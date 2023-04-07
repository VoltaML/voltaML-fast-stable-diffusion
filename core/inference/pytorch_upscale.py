import logging
from typing import List, Optional

import torch
from PIL import Image

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.inference.base_model import InferenceModel
from core.inference.tiled_upscale import StableDiffusionTiledUpscalePipeline
from core.optimizations import optimize_model
from core.schedulers import change_scheduler
from core.types import Job, SDUpscaleQueueEntry
from core.utils import convert_images_to_base64_grid, convert_to_image

logger = logging.getLogger(__name__)


class PyTorchSDUpscaler(InferenceModel):
    "PyTorch Upscaler model for super-resolution."

    def __init__(self, autoload: bool = True, device: str = "cuda"):
        super().__init__(
            model_id="stabilityai/stable-diffusion-x4-upscaler",
            device=device,
        )

        self.pipe: Optional[StableDiffusionTiledUpscalePipeline] = None

        if autoload:
            self.load()

    def load(self):
        pipe = StableDiffusionTiledUpscalePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 if config.api.use_fp32 else torch.float16,
        )

        assert isinstance(pipe, StableDiffusionTiledUpscalePipeline)
        optimize_model(pipe, self.device, config.api.use_fp32)
        self.pipe = pipe

    def unload(self):
        self.pipe = None
        self.memory_cleanup()

    def upscale(self, job: SDUpscaleQueueEntry) -> List[Image.Image]:
        "Upscales an image using the model."

        generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        if job.data.scheduler:
            change_scheduler(
                model=self.pipe,
                scheduler=job.data.scheduler,
            )

        total_images: List[Image.Image] = []

        assert self.pipe is not None

        input_image = convert_to_image(job.data.image)

        for _ in range(job.data.batch_count):
            data = self.pipe(
                prompt=job.data.prompt,
                image=input_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                generator=generator,
                num_images_per_prompt=job.data.batch_size,
                original_image_slice=job.data.original_image_slice,
                tile_border=job.data.tile_border,
                tile_size=job.data.tile_size,
                noise_level=job.data.noise_level,
                # callback=sd_upscale_callback,
            )

            images: List[Image.Image] = [data]
            total_images.extend(images)

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="sd_upscale",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
                },
            )
        )

        return total_images

    def generate(self, job: Job) -> List[Image.Image]:
        if not isinstance(job, SDUpscaleQueueEntry):
            raise TypeError("Expected SDUpscaleQueueEntry")

        images = self.upscale(job)

        self.memory_cleanup()
        return images
