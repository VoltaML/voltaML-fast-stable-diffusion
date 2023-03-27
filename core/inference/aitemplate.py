import logging
import os
from typing import Any, List

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from transformers import CLIPFeatureExtractor
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets.data import Data
from core.files import get_full_model_path
from core.functions import optimize_model
from core.inference.base_model import InferenceModel
from core.inference_callbacks import img2img_callback, txt2img_callback
from core.schedulers import change_scheduler
from core.types import Img2ImgQueueEntry, Job, Txt2ImgQueueEntry
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

logger = logging.getLogger(__name__)


class AITemplateStableDiffusion(InferenceModel):
    "High level wrapper for the AITemplate models"

    def __init__(self, model_id: str, use_f32: bool = False, device: str = "cuda"):
        super().__init__(model_id, use_f32, device)

        self.vae: AutoencoderKL
        self.unet: UNet2DConditionModel
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.scheduler: Any
        self.requires_safety_checker: bool
        self.safety_checker: Any
        self.feature_extractor: CLIPFeatureExtractor

        from aitemplate.compiler import Model  # pylint: disable=E0611,E0401

        self.clip_ait_exe: Model
        self.unet_ait_exe: Model
        self.vae_ait_exe: Model

        self.load()

    def load(self):
        from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

        pipe = StableDiffusionAITPipeline.from_pretrained(
            get_full_model_path(self.model_id),
            torch_dtype=torch.float16,
            directory=os.path.join("data", "aitemplate", self.model_id),
            clip_ait_exe=None,
            unet_ait_exe=None,
            vae_ait_exe=None,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        )
        assert isinstance(pipe, StableDiffusionAITPipeline)
        pipe.to(self.device)
        optimize_model(pipe, "cuda", False, 1)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.requires_safety_checker = False
        self.safety_checker = pipe.safety_checker
        self.feature_extractor = pipe.feature_extractor

        self.clip_ait_exe = pipe.clip_ait_exe
        self.unet_ait_exe = pipe.unet_ait_exe
        self.vae_ait_exe = pipe.vae_ait_exe

    def unload(self):
        del (
            self.vae,
            self.unet,
            self.text_encoder,
            self.tokenizer,
            self.scheduler,
            self.requires_safety_checker,
            self.safety_checker,
            self.clip_ait_exe,
            self.unet_ait_exe,
            self.vae_ait_exe,
        )

        self.memory_cleanup()

    def generate(self, job: Job) -> List[Image.Image]:
        logging.info(f"Adding job {job.data.id} to queue")

        if isinstance(job, Txt2ImgQueueEntry):
            images = self.txt2img(job)
        elif isinstance(job, Img2ImgQueueEntry):
            images = self.img2img(job)
        else:
            raise ValueError("Invalid job type for this model")

        self.memory_cleanup()

        return images

    def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Generates images from text"

        from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

        pipe = StableDiffusionAITPipeline(
            vae=self.vae,
            directory=os.path.join("data", "aitemplate", self.model_id),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
            unet=self.unet,
            feature_extractor=self.feature_extractor,
            clip_ait_exe=self.clip_ait_exe,
            unet_ait_exe=self.unet_ait_exe,
            vae_ait_exe=self.vae_ait_exe,
        )

        generator = torch.Generator(self.device).manual_seed(job.data.seed)

        if job.data.scheduler:
            change_scheduler(
                model=pipe,
                scheduler=job.data.scheduler,
            )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                height=job.data.height,
                width=job.data.width,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=txt2img_callback,
                num_images_per_prompt=job.data.batch_size,
            )
            images: list[Image.Image] = data[0]  # type: ignore

            total_images.extend(images)

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="txt2img",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(total_images),
                },
            )
        )

        return total_images

    def img2img(self, job: Img2ImgQueueEntry) -> List[Image.Image]:
        "Generates images from images"

        from core.aitemplate.src.ait_img2img import StableDiffusionImg2ImgAITPipeline

        pipe = StableDiffusionImg2ImgAITPipeline(
            vae=self.vae,
            directory=os.path.join("data", "aitemplate", self.model_id),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
            unet=self.unet,
            feature_extractor=self.feature_extractor,
            clip_ait_exe=self.clip_ait_exe,
            unet_ait_exe=self.unet_ait_exe,
            vae_ait_exe=self.vae_ait_exe,
        )

        generator = torch.Generator(self.device).manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                init_image=input_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=img2img_callback,
                strength=job.data.strength,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
            )

            images = data[0]
            assert isinstance(images, List)

            total_images.extend(images)

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="img2img",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(total_images),
                },
            )
        )

        return total_images
