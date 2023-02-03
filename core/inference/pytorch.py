import gc
import logging
import os
from typing import Callable, List, Optional

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from PIL import Image
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.inference.unet_tracer import TracedUNet, get_traced_unet
from core.schedulers import change_scheduler
from core.types import Img2ImgQueueEntry, PyTorchModelType, Txt2ImgQueueEntry
from core.utils import convert_image_to_base64, process_image

logger = logging.getLogger(__name__)


class PyTorchInferenceModel:
    "High level model wrapper for PyTorch models"

    def __init__(
        self,
        model_id: str,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 10,
    ) -> None:
        # HuggingFace
        self.model_id: str = model_id
        self.auth: str = auth_token

        # Hardware
        self.use_f32: bool = use_f32
        self.device: str = device

        # Callbacks
        self.callback: Optional[Callable[[int, int, torch.Tensor], None]] = callback
        self.callback_steps: int = callback_steps

        # Components
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.unet: UNet2DConditionModel | TracedUNet

        self.model: PyTorchModelType = self.load()

    def load(self) -> PyTorchModelType:
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

        change_scheduler(
            model=pipe,
            scheduler=KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler,
            config=pipe.config,
        )
        return pipe

    def unload(self) -> None:
        "Unload the model from memory"

        del self.model

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def txt2img(
        self,
        job: Txt2ImgQueueEntry,
    ) -> List[Image.Image]:
        "Generate an image from a prompt"

        if not isinstance(self.model, StableDiffusionPipeline):
            self.model = StableDiffusionPipeline(
                vae=self.model.vae,  # type: ignore
                unet=self.model.unet,  # type: ignore
                text_encoder=self.model.text_encoder,  # type: ignore
                tokenizer=self.model.tokenizer,  # type: ignore
                scheduler=self.model.scheduler,  # type: ignore
                feature_extractor=self.model.feature_extractor,  # type: ignore
                requires_safety_checker=False,
                safety_checker=self.model.safety_checker,  # type: ignore
            )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(
            model=self.model, scheduler=job.scheduler, config=self.model.config
        )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = self.model(
                prompt=job.data.prompt,
                height=job.data.height,
                width=job.data.width,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=self.callback,
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
                    "images": [convert_image_to_base64(i) for i in total_images],
                },
            )
        )

        return total_images

    def img2img(self, job: Img2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image from an image"

        if not isinstance(self.model, StableDiffusionImg2ImgPipeline):
            self.model = StableDiffusionImg2ImgPipeline(
                vae=self.model.vae,  # type: ignore
                unet=self.model.unet,  # type: ignore
                text_encoder=self.model.text_encoder,  # type: ignore
                tokenizer=self.model.tokenizer,  # type: ignore
                scheduler=self.model.scheduler,  # type: ignore
                feature_extractor=self.model.feature_extractor,  # type: ignore
                requires_safety_checker=False,
                safety_checker=self.model.safety_checker,  # type: ignore
            )

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(
            model=self.model, scheduler=job.scheduler, config=self.model.config
        )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            data = self.model(
                prompt=job.data.prompt,
                image=process_image(job.data.image),
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type="pil",
                generator=generator,
                callback=self.callback,
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
                    "images": [convert_image_to_base64(i) for i in total_images],
                },
            )
        )

        return total_images

    def optimize(self) -> None:
        "Optimize the model for inference"

        logger.info("Optimizing model")

        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            self.model.enable_xformers_memory_efficient_attention()
            logger.info("Optimization: Enabled xformers memory efficient attention")
        except ModuleNotFoundError:
            logger.info(
                "Optimization: xformers not available, enabling attention slicing instead"
            )
            self.model.enable_attention_slicing()
            logger.info("Optimization: Enabled attention slicing")

        try:
            self.enable_traced_unet(self.model_id)
            logger.info("Optimization: Enabled traced UNet")
        except ValueError:
            logger.info("Optimization: Traced UNet not available")

        # self.model.vae.enable_vae_slicing()  # type: ignore
        # logger.info("Optimization: Enabled VAE slicing")

        logger.info("Optimization complete")

    def enable_traced_unet(self, model_id: str):
        "Loads a precomputed JIT traced U-Net model."

        traced_unet = get_traced_unet(model_id=model_id, pipe=self.model)
        if traced_unet is not None:
            self.unet = traced_unet
        else:
            raise ValueError(f"Traced U-Net model with id {model_id} does not exist.")
