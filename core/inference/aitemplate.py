import logging
import os
from typing import Any, List, Literal, Optional

import torch
from diffusers import ControlNetModel
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from rich.console import Console
from transformers.models.clip import CLIPFeatureExtractor
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets.data import Data
from core.config import config
from core.functions import init_ait_module
from core.inference.base_model import InferenceModel
from core.inference.functions import load_pytorch_pipeline
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    txt2img_callback,
)
from core.schedulers import change_scheduler
from core.types import (
    Backend,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

logger = logging.getLogger(__name__)
console = Console()


class AITemplateStableDiffusion(InferenceModel):
    "High level wrapper for the AITemplate models"

    def __init__(
        self,
        model_id: str,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        device: str = "cuda",
    ):
        super().__init__(model_id, device)

        self.backend: Backend = "AITemplate"

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
        self.current_controlnet: str = ""

        if "__dynamic" in self.model_id:
            self.load_dynamic()
        else:
            self.load()

    @property
    def directory(self) -> str:
        "Directory where the model is stored"

        return os.path.join("data", "aitemplate", self.model_id)

    def load(self):
        from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

        pipe = load_pytorch_pipeline(
            self.model_id,
            device=self.device,
            is_for_aitemplate=True,
        )

        pipe.to(self.device)

        pipe.unet = None  # type: ignore
        self.memory_cleanup()

        with console.status("[bold green]Loading AITemplate model..."):
            pipe = StableDiffusionAITPipeline(
                unet=pipe.unet,  # type: ignore
                vae=pipe.vae,  # type: ignore
                text_encoder=pipe.text_encoder,  # type: ignore
                tokenizer=pipe.tokenizer,  # type: ignore
                scheduler=pipe.scheduler,  # type: ignore
                directory=self.directory,
                clip_ait_exe=None,
                unet_ait_exe=None,
                vae_ait_exe=None,
                requires_safety_checker=False,
                safety_checker=None,  # type: ignore
                feature_extractor=None,  # type: ignore
            )
        assert isinstance(pipe, StableDiffusionAITPipeline)

        self.unet = pipe.unet  # type: ignore
        self.vae = pipe.vae
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
        self.type: Literal["static", "dynamic"] = "static"

    def load_dynamic(self):
        from core.aitemplate.src.dynamic_ait_txt2img import (
            StableDiffusionDynamicAITPipeline,
        )

        pipe = load_pytorch_pipeline(
            self.model_id,
            device=self.device,
            is_for_aitemplate=True,
        )

        pipe.to(self.device)

        self.memory_cleanup()

        with console.status("[bold green]Loading AITemplate model..."):
            pipe = StableDiffusionDynamicAITPipeline(
                vae=pipe.vae,  # type: ignore
                text_encoder=pipe.text_encoder,  # type: ignore
                tokenizer=pipe.tokenizer,  # type: ignore
                scheduler=pipe.scheduler,  # type: ignore
                unet=pipe.unet,  # type: ignore
                directory=self.directory,
                clip_ait_exe=None,
                unet_ait_exe=None,
                vae_ait_exe=None,
                requires_safety_checker=False,
                safety_checker=None,  # type: ignore
                feature_extractor=None,  # type: ignore
            )
        assert isinstance(pipe, StableDiffusionDynamicAITPipeline)

        self.vae = pipe.vae  # type: ignore
        self.text_encoder = pipe.text_encoder  # type: ignore
        self.unet = pipe.unet  # type: ignore
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.requires_safety_checker = False
        self.safety_checker = pipe.safety_checker  # type: ignore
        self.feature_extractor = pipe.feature_extractor  # type: ignore

        self.clip_ait_exe = pipe.clip_ait_exe
        self.unet_ait_exe = pipe.unet_ait_exe
        self.vae_ait_exe = pipe.vae_ait_exe

        self.current_unet: Literal["unet", "controlnet_unet"] = "unet"
        self.type = "dynamic"

    def unload(self):
        for property_ in (
            "vae",
            "text_encoder",
            "unet",
            "tokenizer",
            "scheduler",
            "safety_checker",
            "clip_ait_exe",
            "unet_ait_exe",
            "vae_ait_exe",
            "controlnet",
        ):
            if hasattr(self, property_):
                del self.__dict__[property_]

        self.memory_cleanup()

    def manage_optional_components(
        self,
        *,
        target_controlnet: str = "",
    ) -> None:
        "Cleanup old components"

        logger.debug(
            f"Current controlnet: {self.current_controlnet}, target: {target_controlnet}"
        )

        if self.current_controlnet != target_controlnet:
            # Cleanup old controlnet
            self.controlnet = None
            self.memory_cleanup()

            if not target_controlnet:
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

                self.current_controlnet = target_controlnet
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
                target_controlnet,
                resume_download=True,
                torch_dtype=config.api.dtype,
                use_auth_token=self.auth,
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
            self.current_controlnet = target_controlnet

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

    def txt2img(
        self,
        job: Txt2ImgQueueEntry,
    ) -> List[Image.Image]:
        "Generates images from text"

        if self.type == "static":
            from core.aitemplate.src.ait_txt2img import StableDiffusionAITPipeline

            cls = StableDiffusionAITPipeline
        else:
            from core.aitemplate.src.dynamic_ait_txt2img import (
                StableDiffusionDynamicAITPipeline,
            )

            cls = StableDiffusionDynamicAITPipeline

        self.manage_optional_components()

        pipe = cls(
            unet=self.unet,
            vae=self.vae,
            directory=self.directory,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
                },
            )
        )

        return total_images

    def img2img(
        self,
        job: Img2ImgQueueEntry,
    ) -> List[Image.Image]:
        "Generates images from images"

        from core.aitemplate.src.ait_img2img import StableDiffusionImg2ImgAITPipeline

        self.manage_optional_components()

        pipe = StableDiffusionImg2ImgAITPipeline(
            unet=self.unet,
            vae=self.vae,
            directory=self.directory,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
                },
            )
        )

        return total_images

    def controlnet2img(
        self,
        job: ControlNetQueueEntry,
    ) -> List[Image.Image]:
        "Generates images from images"

        self.manage_optional_components(target_controlnet=job.data.controlnet)

        assert self.controlnet is not None

        from core.aitemplate.src.ait_controlnet import (
            StableDiffusionControlNetAITPipeline,
        )

        pipe = StableDiffusionControlNetAITPipeline(
            unet=self.unet,
            vae=self.vae,
            directory=self.directory,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            requires_safety_checker=self.requires_safety_checker,
            controlnet=self.controlnet,
            feature_extractor=self.feature_extractor,
            clip_ait_exe=self.clip_ait_exe,
            unet_ait_exe=self.unet_ait_exe,
            vae_ait_exe=self.vae_ait_exe,
        )

        generator = torch.Generator(self.device).manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        from core.controlnet_preprocessing import image_to_controlnet_input

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        # Preprocess the image if needed
        if not job.data.is_preprocessed:
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
                },
            )
        )

        return total_images
