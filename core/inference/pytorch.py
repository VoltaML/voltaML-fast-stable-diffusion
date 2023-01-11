import gc
import logging
import os
from typing import Callable, List, Optional

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from PIL.Image import Image

from core.pytorch.schedulers import change_scheduler
from core.types import Scheduler, Txt2imgData

os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"] = "1"


class PyTorchInferenceModel:
    "High level model wrapper for PyTorch models"

    def __init__(
        self,
        model_id: str,
        scheduler: Scheduler = Scheduler.default,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
    ) -> None:
        self.use_f32: bool = use_f32
        self.auth: str = auth_token
        self.model_id_or_path: str = model_id
        self.device: str = device
        self.callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
        self.callback_steps: int = 10
        self.model: Optional[StableDiffusionPipeline] = self.load()
        change_scheduler(model=self.model, scheduler=scheduler)

    def load(self) -> StableDiffusionPipeline:
        "Load the model from HuggingFace"

        logging.info(
            f"Loading {self.model_id_or_path} with {'f32' if self.use_f32 else 'f16'}"
        )

        
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch.float16 if self.use_f32 else torch.float32,
            use_auth_token=self.auth,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=os.getcwd() + "/huggingface_models/"
        )
        
        assert isinstance(pipe, StableDiffusionPipeline)
        return pipe.to(self.device)

    def unload(self) -> None:
        "Unload the model from memory"

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def generate(self, job: Txt2imgData, scheduler: Scheduler) -> List[Image]:
        "Generate an image from a prompt"

        if self.model is None:
            raise ValueError("Model not loaded")

        generator = torch.Generator("cuda").manual_seed(job.seed)

        change_scheduler(model=self.model, scheduler=scheduler)

        data = self.model(
            prompt=job.prompt,
            height=job.height,
            width=job.width,
            num_inference_steps=job.steps,
            guidance_scale=job.guidance_scale,
            negative_prompt=job.negative_prompt,
            output_type="pil",
            generator=generator,
            return_dict=False,
            callback=self.callback,
            callback_steps=self.callback_steps,
        )

        images: list[Image] = data[0]

        return images

    def optimize(self, enable_cpu_offload: bool = False) -> None:
        "Optimize the model for inference"

        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.enable_attention_slicing()
        logging.info("Optimization: Enabled attention slicing")

        try:
            self.model.enable_vae_slicing()
            logging.info("Optimization: Enabled VAE slicing")
        except AttributeError:
            logging.info("Optimization: VAE slicing not available")

        if enable_cpu_offload:
            try:
                self.model.enable_sequential_cpu_offload()
                logging.info("Optimization: Enabled sequential CPU offload")
            except ModuleNotFoundError:
                logging.info("Optimization: accelerate not available")

        try:
            self.model.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            logging.info("Optimization: xformers not available")
