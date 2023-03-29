import logging
import os
from typing import Any, List, Literal, Optional

import torch
from diffusers import ControlNetModel
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from transformers import CLIPFeatureExtractor
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets.data import Data
from core.config import config
from core.controlnet import image_to_controlnet_input
from core.files import get_full_model_path
from core.functions import init_ait_module, optimize_model
from core.inference.base_model import InferenceModel
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    txt2img_callback,
)
from core.schedulers import change_scheduler
from core.types import (
    ControlNetMode,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

logger = logging.getLogger(__name__)


class AITemplateStableDiffusion(InferenceModel):
    "High level wrapper for the AITemplate models"

    def __init__(
        self,
        model_id: str,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
    ):
        super().__init__(model_id, use_f32, device)

        # HuggingFace auth token
        self.auth = auth_token

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

        self.controlnet: Optional[ControlNetModel] = None
        self.current_controlnet: ControlNetMode = ControlNetMode.NONE

        self.load()

    @property
    def directory(self) -> str:
        "Directory where the model is stored"

        return os.path.join("data", "aitemplate", self.model_id)

    def load(self):
        from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

        pipe = StableDiffusionAITPipeline.from_pretrained(
            get_full_model_path(self.model_id),
            torch_dtype=torch.float16,
            directory=self.directory,
            clip_ait_exe=None,
            unet_ait_exe=None,
            vae_ait_exe=None,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        )
        assert isinstance(pipe, StableDiffusionAITPipeline)
        pipe.to(self.device)

        # Disable optLevel for AITemplate models and optimize the model
        _opt = config.api.optLevel
        config.api.optLevel = 1
        optimize_model(pipe, self.device, False)
        config.api.optLevel = _opt

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

        self.current_unet: Literal["unet", "controlnet_unet"] = "unet"

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

    def manage_optional_components(
        self,
        *,
        target_controlnet: ControlNetMode = ControlNetMode.NONE,
    ) -> None:
        "Cleanup old components"

        if self.current_controlnet != target_controlnet:
            # Cleanup old controlnet
            self.controlnet = None
            self.memory_cleanup()

            if target_controlnet == ControlNetMode.NONE:
                # Load basic unet if requested

                if self.current_unet == "controlnet_unet":
                    logger.info("Loading basic unet")

                    del self.unet_ait_exe

                    self.memory_cleanup()

                    self.unet_ait_exe = init_ait_module(
                        model_name="UNet2DConditionModel", workdir=self.directory
                    )
                    self.current_unet = (  # pylint: disable=attribute-defined-outside-init
                        "unet"
                    )

                    logger.info("Done loading basic unet")
                return
            else:
                # Load controlnet unet if requested

                if self.current_unet == "unet":
                    logger.info("Loading controlnet unet")

                    del self.unet_ait_exe

                    self.memory_cleanup()

                    self.unet_ait_exe = init_ait_module(
                        model_name="ControlNetUNet2DConditionModel",
                        workdir=self.directory,
                    )
                    self.current_unet = (  # pylint: disable=attribute-defined-outside-init
                        "controlnet_unet"
                    )

                    logger.info("Done loading controlnet unet")

            # Load new controlnet if needed
            cn = ControlNetModel.from_pretrained(
                target_controlnet.value,
                resume_download=True,
                torch_dtype=torch.float32 if self.use_f32 else torch.float16,
                use_auth_token=self.auth,
                cache_dir=config.api.cache_dir,
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
        self.memory_cleanup()

    def generate(self, job: Job) -> List[Image.Image]:
        logging.info(f"Adding job {job.data.id} to queue")

        if isinstance(job, Txt2ImgQueueEntry):
            images = self.txt2img(job)
        elif isinstance(job, Img2ImgQueueEntry):
            images = self.img2img(job)
        elif isinstance(job, ControlNetQueueEntry):
            images = self.controlnet2img(job)
        else:
            raise ValueError("Invalid job type for this model")

        self.memory_cleanup()

        return images

    def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Generates images from text"

        from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

        self.manage_optional_components()

        pipe = StableDiffusionAITPipeline(
            vae=self.vae,
            directory=self.directory,
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

        self.manage_optional_components()

        pipe = StableDiffusionImg2ImgAITPipeline(
            vae=self.vae,
            directory=self.directory,
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

    def controlnet2img(self, job: ControlNetQueueEntry) -> List[Image.Image]:
        "Generates images from images"

        self.manage_optional_components(target_controlnet=job.data.controlnet)

        assert self.controlnet is not None

        from core.aitemplate.src.ait_controlnet import (
            StableDiffusionControlNetAITPipeline,
        )

        pipe = StableDiffusionControlNetAITPipeline(
            vae=self.vae,
            directory=self.directory,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
            unet=self.unet,
            controlnet=self.controlnet,
            feature_extractor=self.feature_extractor,
            clip_ait_exe=self.clip_ait_exe,
            unet_ait_exe=self.unet_ait_exe,
            vae_ait_exe=self.vae_ait_exe,
        )

        generator = torch.Generator(self.device).manual_seed(job.data.seed)

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

            total_images.extend(images)

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
