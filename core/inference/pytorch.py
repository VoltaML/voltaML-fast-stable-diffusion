import logging
import os
from typing import List, Optional

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
    ) -> None:
        self.use_f32 = use_f32
        self.auth = auth_token
        self.model_id = model_id
        self.model: Optional[StableDiffusionPipeline] = self.load()
        change_scheduler(model=self.model, scheduler=scheduler)

    def load(self) -> StableDiffusionPipeline:
        "Load the model from HuggingFace"

        logging.info(f"Loading {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.use_f32 else torch.float32,
            use_auth_token=self.auth,
            safety_checker=None,
        )

        assert isinstance(pipe, StableDiffusionPipeline)
        return pipe.to("cuda")

    def unload(self) -> None:
        "Unload the model from memory"

        self.model = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
        )

        images: list[Image] = data[0]

        return images

    def optimize(self) -> None:
        "Optimize the model for inference"

        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.enable_attention_slicing()
        if hasattr(self.model, "enable_vae_slicing"):
            self.model.enable_vae_slicing()
