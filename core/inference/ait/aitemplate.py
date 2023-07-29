import logging
import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

import torch
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from PIL import Image
from rich.console import Console
from transformers.models.clip import CLIPFeatureExtractor
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets.data import Data
from core.config import config
from core.inference.ait.pipeline import StableDiffusionAITPipeline
from core.inference.base_model import InferenceModel
from core.inference.functions import load_pytorch_pipeline
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    txt2img_callback,
)
from core.types import (
    Backend,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

from ..utilities.aitemplate import init_ait_module
from ..utilities.controlnet import image_to_controlnet_input
from ..utilities.lwp import get_weighted_text_embeddings
from ..utilities.scheduling import change_scheduler

if TYPE_CHECKING:
    from diffusers.pipelines import StableDiffusionPipeline


logger = logging.getLogger(__name__)
console = Console()


class AITemplateStableDiffusion(InferenceModel):
    "High level wrapper for the AITemplate models"

    def __init__(
        self,
        model_id: str,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(model_id, device)

        self.backend: Backend = "AITemplate"

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

        self.load()

    @property
    def directory(self) -> str:
        "Directory where the model is stored"

        return os.path.join("data", "aitemplate", self.model_id)

    def load(self):
        # pylint: disable=redefined-outer-name,reimported
        from .pipeline import StableDiffusionAITPipeline

        pipe = load_pytorch_pipeline(
            self.model_id,
            device=self.device,
            is_for_aitemplate=True,
        )

        pipe.to(self.device)
        self.memory_cleanup()

        with console.status("[bold green]Loading AITemplate model..."):
            pipe = StableDiffusionAITPipeline(
                unet=pipe.unet,  # type: ignore
                vae=pipe.vae,  # type: ignore
                controlnet=self.controlnet,
                text_encoder=pipe.text_encoder,  # type: ignore
                tokenizer=pipe.tokenizer,  # type: ignore
                scheduler=pipe.scheduler,  # type: ignore
                directory=self.directory,
                clip_ait_exe=None,
                unet_ait_exe=None,
                vae_ait_exe=None,
            )
        assert isinstance(pipe, StableDiffusionAITPipeline)

        self.unet = pipe.unet  # type: ignore
        # self.unet.cpu()
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        # self.text_encoder.cpu()
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.requires_safety_checker = False
        self.clip_ait_exe = pipe.clip_ait_exe
        self.unet_ait_exe = pipe.unet_ait_exe
        self.vae_ait_exe = pipe.vae_ait_exe

        self.current_unet: Literal["unet", "controlnet_unet"] = "unet"
        self.memory_cleanup()

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
                    # self.unet.to(config.api.device)

                    self.unet_ait_exe = init_ait_module(
                        model_name="UNet2DConditionModel", workdir=self.directory
                    )
                    from core.aitemplate.src.modeling import mapping

                    self.unet_ait_exe.set_many_constants_with_tensors(
                        mapping.map_unet(self.unet)
                    )
                    self.unet_ait_exe.fold_constants()
                    self.current_unet = (  # pylint: disable=attribute-defined-outside-init
                        "unet"
                    )

                    # self.unet.cpu()

                    logger.info("Done loading basic unet")

                self.current_controlnet = target_controlnet
                return
            else:
                # Load controlnet unet if requested

                if self.current_unet == "unet":
                    logger.info("Loading controlnet unet")

                    del self.unet_ait_exe

                    self.memory_cleanup()
                    # self.unet.to(config.api.device, config.api.dtype)

                    self.unet_ait_exe = init_ait_module(
                        model_name="ControlNetUNet2DConditionModel",
                        workdir=self.directory,
                    )
                    from core.aitemplate.src.modeling import mapping

                    self.unet_ait_exe.set_many_constants_with_tensors(
                        mapping.map_unet(self.unet)
                    )
                    self.unet_ait_exe.fold_constants()
                    self.current_unet = (  # pylint: disable=attribute-defined-outside-init
                        "controlnet_unet"
                    )

                    # self.unet.cpu()

                    logger.info("Done loading controlnet unet")

            # Load new controlnet if needed
            cn = ControlNetModel.from_pretrained(
                target_controlnet,
                resume_download=True,
                torch_dtype=config.api.dtype,
            )

            assert isinstance(cn, ControlNetModel)
            try:
                cn.enable_xformers_memory_efficient_attention()
                logger.info("Optimization: Enabled xformers memory efficient attention")
            except ModuleNotFoundError:
                logger.info(
                    "Optimization: xformers not available, enabling attention slicing instead"
                )

            cn.to(device=torch.device(self.device), dtype=config.api.dtype)
            self.controlnet = cn
            self.current_controlnet = target_controlnet

        # Clean memory
        self.memory_cleanup()

    def create_pipe(
        self,
        controlnet: str = "",
        seed: int = -1,
        scheduler: Optional[Tuple[Any, bool]] = None,
    ) -> Tuple["StableDiffusionAITPipeline", torch.Generator]:
        "Centralized way to create new pipelines."

        self.manage_optional_components(target_controlnet=controlnet)

        # pylint: disable=redefined-outer-name,reimported
        from .pipeline import StableDiffusionAITPipeline

        pipe = StableDiffusionAITPipeline(
            unet=self.unet,
            vae=self.vae,
            directory=self.directory,
            controlnet=self.controlnet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            clip_ait_exe=self.clip_ait_exe,
            unet_ait_exe=self.unet_ait_exe,
            vae_ait_exe=self.vae_ait_exe,
        )

        generator = torch.Generator(self.device).manual_seed(seed)

        if scheduler:
            change_scheduler(
                model=pipe,
                scheduler=scheduler[0],
                use_karras_sigmas=scheduler[1],
            )
        return pipe, generator

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
        pipe, generator = self.create_pipe(
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                pipe, job.data.prompt, job.data.negative_prompt
            )
            data = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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
        pipe, generator = self.create_pipe(
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                pipe, job.data.prompt, job.data.negative_prompt
            )
            data = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=input_image,  # type: ignore
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=img2img_callback,
                strength=job.data.strength,  # type: ignore
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
        pipe, generator = self.create_pipe(
            controlnet=job.data.controlnet,
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        # Preprocess the image if needed
        if not job.data.is_preprocessed:
            input_image = image_to_controlnet_input(input_image, job.data)

        total_images: List[Image.Image] = [input_image]

        for _ in range(job.data.batch_count):
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                pipe, job.data.prompt, job.data.negative_prompt
            )
            data = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=input_image,  # type: ignore
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=controlnet_callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                controlnet_conditioning_scale=job.data.controlnet_conditioning_scale,  # type: ignore
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
