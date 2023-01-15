import gc
import logging
import os
from typing import Callable, Dict, List, Optional

import torch
from PIL.Image import Image

from core.config import config
from core.diffusers.kdiffusion import StableDiffusionKDiffusionPipeline
from core.schedulers import change_scheduler
from core.types import KDiffusionScheduler, Txt2imgData


class PyTorchInferenceModel:
    "High level model wrapper for PyTorch models"

    def __init__(
        self,
        model_id: str,
        scheduler: KDiffusionScheduler = KDiffusionScheduler.euler_a,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
        callback: Optional[Callable[[Dict], None]] = None,
        callback_steps: int = 10,
    ) -> None:
        self.use_f32: bool = use_f32
        self.auth: str = auth_token
        self.model_id_or_path: str = model_id
        self.device: str = device
        self.callback: Optional[Callable[[Dict], None]] = callback
        self.callback_steps: int = callback_steps
        self.model: Optional[StableDiffusionKDiffusionPipeline] = self.load()
        change_scheduler(model=self.model, scheduler=scheduler)

    def load(self) -> StableDiffusionKDiffusionPipeline:
        "Load the model from HuggingFace"

        logging.info(
            f"Loading {self.model_id_or_path} with {'f32' if self.use_f32 else 'f16'}"
        )

        pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_id_or_path,
            torch_dtype=torch.float16 if self.use_f32 else torch.float32,
            use_auth_token=self.auth,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            cache_dir=config.cache_dir,
        )

        assert isinstance(pipe, StableDiffusionKDiffusionPipeline)
        pipe = pipe.to(self.device)
        pipe.set_scheduler("sample_euler_ancestral")
        return pipe

    def unload(self) -> None:
        "Unload the model from memory"

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def generate(
        self,
        job: Txt2imgData,
        scheduler: KDiffusionScheduler,
        use_karras_sigmas: bool = True,
    ) -> List[Image]:
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
            use_karras_sigmas=use_karras_sigmas,
        )

        images: list[Image] = data[0]

        return images

    def optimize(self, enable_cpu_offload: bool = False) -> None:
        "Optimize the model for inference"

        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.enable_attention_slicing()
        logging.info("Optimization: Enabled attention slicing")

        if enable_cpu_offload:
            try:
                self.model.enable_sequential_cpu_offload()
                logging.info("Optimization: Enabled sequential CPU offload")
            except ModuleNotFoundError:
                logging.info("Optimization: accelerate not available")

        try:
            self.model.enable_xformers_memory_efficient_attention()
            logging.info("Optimization: Enabled xformers memory efficient attention")
        except ModuleNotFoundError:
            logging.info("Optimization: xformers not available")
