import gc
import logging
import os
from typing import Any, List, Union

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from PIL import Image, ImageOps
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.functions import img2img_callback, inpaint_callback, txt2img_callback
from core.inference.base_model import InferenceModel
from core.inference.unet_tracer import TracedUNet, get_traced_unet
from core.schedulers import change_scheduler
from core.types import Img2ImgQueueEntry, InpaintQueueEntry, Txt2ImgQueueEntry
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

logger = logging.getLogger(__name__)


class PyTorchStableDiffusion(InferenceModel):
    "High level model wrapper for PyTorch models"

    def __init__(
        self,
        model_id: str,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(model_id, use_f32, device)

        # HuggingFace
        self.auth: str = auth_token

        # Components
        self.vae: AutoencoderKL
        self.unet: Union[UNet2DConditionModel, TracedUNet]
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.scheduler: Any
        self.feature_extractor: Any
        self.requires_safety_checker: bool
        self.safety_checker: Any

        self.load()

    def load(self):
        "Load the model from HuggingFace"

        logger.info(f"Loading {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.float32 if self.use_f32 else torch.float16,
            use_auth_token=self.auth,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            cache_dir=config.cache_dir,
        )

        logger.debug(f"Loaded {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        assert isinstance(pipe, StableDiffusionPipeline)
        pipe = pipe.to(self.device)

        self.optimize(pipe)

        self.vae = pipe.vae  # type: ignore
        self.unet = pipe.unet  # type: ignore
        self.text_encoder = pipe.text_encoder  # type: ignore
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        self.feature_extractor = pipe.feature_extractor  # type: ignore
        self.requires_safety_checker = False  # type: ignore
        self.safety_checker = pipe.safety_checker  # type: ignore

    def unload(self) -> None:
        "Unload the model from memory"

        del (
            self.vae,
            self.unet,
            self.text_encoder,
            self.tokenizer,
            self.scheduler,
            self.feature_extractor,
            self.requires_safety_checker,
            self.safety_checker,
            self.image_encoder,
        )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def cleanup_old_components(self, keep_variations: bool = False) -> None:
        "Cleanup old components"

        if not keep_variations:
            self.image_encoder = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def txt2img(
        self,
        job: Txt2ImgQueueEntry,
    ) -> List[Image.Image]:
        "Generate an image from a prompt"

        self.cleanup_old_components()

        pipe = StableDiffusionPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
        )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

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
            images: list[Image.Image] = data[0]

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
        "Generate an image from an image"

        self.cleanup_old_components()

        pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
        )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                image=input_image,
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

    def inpaint(self, job: InpaintQueueEntry) -> List[Image.Image]:
        "Generate an image from an image"

        self.cleanup_old_components()

        pipe = StableDiffusionInpaintPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
        )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        input_image = convert_to_image(job.data.image).convert("RGB")
        input_image = resize(input_image, job.data.width, job.data.height)

        input_mask_image = convert_to_image(job.data.mask_image).convert("RGB")
        input_mask_image = ImageOps.invert(input_mask_image)
        input_mask_image = resize(input_mask_image, job.data.height, job.data.width)
        input_mask_image.save("mask.png")

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                image=input_image,
                mask_image=input_mask_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=inpaint_callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
            )

            images = data[0]
            assert isinstance(images, List)

            total_images.extend(images)

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="inpainting",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(total_images),
                },
            )
        )

        return total_images

    def optimize(self, pipe: StableDiffusionPipeline) -> None:
        "Optimize the model for inference"

        logger.info("Optimizing model")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Optimization: Enabled xformers memory efficient attention")
        except ModuleNotFoundError:
            logger.info(
                "Optimization: xformers not available, enabling attention slicing instead"
            )
            pipe.enable_attention_slicing()
            logger.info("Optimization: Enabled attention slicing")

        try:
            self.enable_traced_unet(self.model_id, pipe.unet)  # type: ignore
            logger.info("Optimization: Enabled traced UNet")
        except ValueError:
            logger.info("Optimization: Traced UNet not available")

        logger.info("Optimization complete")

    def enable_traced_unet(self, model_id: str, unet: UNet2DConditionModel):
        "Loads a precomputed JIT traced U-Net model."

        traced_unet = get_traced_unet(model_id=model_id, untraced_unet=unet)
        if traced_unet is not None:
            self.unet = traced_unet
        else:
            raise ValueError(f"Traced U-Net model with id {model_id} does not exist.")

    def generate(
        self,
        job: Union[
            Txt2ImgQueueEntry,
            Img2ImgQueueEntry,
            InpaintQueueEntry,
        ],
    ):
        "Generate images from the queue"

        logging.info(f"Adding job {job.data.id} to queue")

        if isinstance(job, Txt2ImgQueueEntry):
            images = self.txt2img(job)
        elif isinstance(job, Img2ImgQueueEntry):
            images = self.img2img(job)
        elif isinstance(job, InpaintQueueEntry):
            images = self.inpaint(job)

        return images

    def save(self, path: str = "converted"):
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
        )

        pipe.save_pretrained(path)
