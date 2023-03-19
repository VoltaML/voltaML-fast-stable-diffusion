import logging
import os
from typing import Any, List, Optional

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image, ImageOps
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.controlnet import image_to_controlnet_input
from core.files import get_full_model_path
from core.functions import optimize_model
from core.inference.base_model import InferenceModel
from core.inference.LPW_SD import StableDiffusionLongPromptWeightingPipeline
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    inpaint_callback,
    txt2img_callback,
)
from core.schedulers import change_scheduler
from core.types import (
    ControlNetMode,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
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
        autoload: bool = True,
    ) -> None:
        super().__init__(model_id, use_f32, device)

        # HuggingFace
        self.auth: str = auth_token

        # Components
        self.vae: AutoencoderKL
        self.unet: UNet2DConditionModel
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.scheduler: Any
        self.feature_extractor: Any
        self.requires_safety_checker: bool
        self.safety_checker: Any
        self.image_encoder: Any
        self.controlnet: Optional[ControlNetModel]

        self.current_controlnet: ControlNetMode = ControlNetMode.NONE

        if autoload:
            self.load()

    def load(self):
        "Load the model from HuggingFace"

        logger.info(f"Loading {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
            pretrained_model_name_or_path=get_full_model_path(self.model_id),
            torch_dtype=torch.float32 if self.use_f32 else torch.float16,
            use_auth_token=self.auth,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            cache_dir=config.cache_dir,
        )

        logger.debug(f"Loaded {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        assert isinstance(pipe, StableDiffusionLongPromptWeightingPipeline)
        pipe.to(self.device)

        optimize_model(pipe)

        self.vae = pipe.vae  # type: ignore
        self.unet = pipe.unet  # type: ignore
        self.text_encoder = pipe.text_encoder  # type: ignore
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        self.feature_extractor = pipe.feature_extractor  # type: ignore
        self.requires_safety_checker = False  # type: ignore
        self.safety_checker = pipe.safety_checker  # type: ignore

        self.cleanup()

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
        )

        if hasattr(self, "image_encoder"):
            if self.image_encoder is not None:
                del self.image_encoder

        if hasattr(self, "controlnet"):
            if self.controlnet is not None:
                del self.controlnet

        self.cleanup()

    def manage_optional_components(
        self,
        *,
        variations: bool = False,
        target_controlnet: ControlNetMode = ControlNetMode.NONE,
    ) -> None:
        "Cleanup old components"

        if not variations:
            self.image_encoder = None

        if self.current_controlnet != target_controlnet:
            # Cleanup old controlnet
            self.controlnet = None
            self.cleanup()

            if target_controlnet == ControlNetMode.NONE:
                return

            # Load new controlnet if needed
            cn = ControlNetModel.from_pretrained(
                target_controlnet.value,
                resume_download=True,
                torch_dtype=torch.float32 if self.use_f32 else torch.float16,
                use_auth_token=self.auth,
                cache_dir=config.cache_dir,
            )

            assert isinstance(cn, ControlNetModel)
            try:
                cn.enable_xformers_memory_efficient_attention()
                logger.info("Optimization: Enabled xformers memory efficient attention")
            except ModuleNotFoundError:
                logger.info(
                    "Optimization: xformers not available, enabling attention slicing instead"
                )

            cn.to(self.device)
            self.controlnet = cn

        # Clean memory
        self.cleanup()

    def txt2img(
        self,
        job: Txt2ImgQueueEntry,
    ) -> List[Image.Image]:
        "Generate an image from a prompt"

        self.manage_optional_components()

        pipe = StableDiffusionLongPromptWeightingPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
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
            data = pipe.text2img(
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
        "Generate an image from an image"

        self.manage_optional_components()

        pipe = StableDiffusionLongPromptWeightingPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            safety_checker=self.safety_checker,
        )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = pipe.img2img(
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

            if not data:
                raise ValueError("No data returned from pipeline")

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

        self.manage_optional_components()

        pipe = StableDiffusionLongPromptWeightingPipeline(
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
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
            data = pipe.inpaint(
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

            if not data:
                raise ValueError("No data returned from pipeline")

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

    def controlnet2img(self, job: ControlNetQueueEntry) -> List[Image.Image]:
        "Generate an image from an image and controlnet conditioning"

        self.manage_optional_components(target_controlnet=job.data.controlnet)

        assert self.controlnet is not None

        pipe = StableDiffusionControlNetPipeline(
            controlnet=self.controlnet,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
            scheduler=self.scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            vae=self.vae,
        )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        input_image = image_to_controlnet_input(input_image, job.data)

        total_images: List[Image.Image] = [input_image]

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                image=input_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=controlnet_callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                controlnet_conditioning_scale=job.data.controlnet_conditioning_scale,
                height=job.data.height,
                width=job.data.width,
            )

            images = data[0]
            assert isinstance(images, List)

            total_images.extend(images)  # type: ignore

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="controlnet",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(total_images),
                },
            )
        )

        return total_images

    def generate(
        self,
        job: Job,
    ):
        "Generate images from the queue"

        logging.info(f"Adding job {job.data.id} to queue")

        if isinstance(job, Txt2ImgQueueEntry):
            images = self.txt2img(job)
        elif isinstance(job, Img2ImgQueueEntry):
            images = self.img2img(job)
        elif isinstance(job, InpaintQueueEntry):
            images = self.inpaint(job)
        elif isinstance(job, ControlNetQueueEntry):
            images = self.controlnet2img(job)
        else:
            raise ValueError("Invalid job type for this pipeline")

        # Clean memory and return images
        self.cleanup()
        return images

    def save(self, path: str = "converted"):
        "Dump current pipeline to specified path"

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
