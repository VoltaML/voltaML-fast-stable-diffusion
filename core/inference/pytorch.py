import os
from typing import List, Literal, Optional

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from PIL.Image import Image

from api.types import Txt2imgJob

os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"] = "1"


class PyTorchInferenceModel:
    def __init__(
        self,
        model_id: str,
        scheduler: Literal["euler_a", "euler", "ddim", None] = None,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
    ) -> None:
        self.use_f32 = use_f32
        self.auth = auth_token
        self.model_id = model_id
        self.scheduler = self.get_scheduler(scheduler)
        self.model: Optional[StableDiffusionPipeline] = self.load()

    def load(self) -> StableDiffusionPipeline:
        print(f"Loading model with {'f32' if self.use_f32 else 'f16'}")
        if self.scheduler:
            return StableDiffusionPipeline.from_pretrained(  # type: ignore
                self.model_id,
                torch_dtype=torch.float32 if self.use_f32 else torch.float16,
                scheduler=self.scheduler,
                use_auth_token=self.auth,
                safety_checker=None,
            ).to(  # type: ignore
                "cuda"
            )
        else:
            return StableDiffusionPipeline.from_pretrained(  # type: ignore
                self.model_id,
                torch_dtype=torch.float32 if self.use_f32 else torch.float16,
                use_auth_token=self.auth,
                safety_checker=None,
            ).to(  # type: ignore
                "cuda"
            )

    def get_scheduler(self, scheduler: Optional[str]):
        if scheduler == "euler_a":
            return EulerAncestralDiscreteScheduler.from_config(
                self.model_id, subfolder="scheduler"
            )
        elif scheduler == "euler":
            return EulerDiscreteScheduler.from_config(
                self.model_id, subfolder="scheduler"
            )
        elif scheduler == "ddim":
            return DDIMScheduler.from_config(self.model_id, subfolder="scheduler")
        elif not scheduler:
            return None
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

    def unload(self) -> None:
        self.model = None

    def generate(self, job: Txt2imgJob) -> List[Image]:
        if self.model is None:
            raise ValueError("Model not loaded")

        generator = torch.Generator("cuda").manual_seed(job.seed)

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
