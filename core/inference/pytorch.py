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
from core.flags import HighResFixFlag
from core.inference.base_model import InferenceModel
from core.inference.functions import load_pytorch_pipeline
from core.inference.latents import scale_latents
from core.inference.lwp import get_weighted_text_embeddings
from core.inference.lwp_sd import StableDiffusionLongPromptWeightingPipeline
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    inpaint_callback,
    txt2img_callback,
)
from core.lora import load_safetensors_loras
from core.schedulers import change_scheduler
from core.types import (
    Backend,
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
        device: str = "cuda",
        autoload: bool = True,
    ) -> None:
        super().__init__(model_id, device)

        self.backend: Backend = "PyTorch"

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

        self.loras: List[str] = []

        if autoload:
            self.load()

    def load(self):
        "Load the model from HuggingFace"

        logger.info(
            f"Loading {self.model_id} with {'f32' if config.api.use_fp32 else 'f16'}"
        )

        pipe = load_pytorch_pipeline(
            self.model_id,
            auth=self.auth,
            device=self.device,
        )

        self.vae = pipe.vae  # type: ignore
        self.unet = pipe.unet  # type: ignore
        self.text_encoder = pipe.text_encoder  # type: ignore
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        self.feature_extractor = pipe.feature_extractor  # type: ignore
        self.requires_safety_checker = False  # type: ignore
        self.safety_checker = pipe.safety_checker  # type: ignore

        del pipe

        self.memory_cleanup()

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

        self.memory_cleanup()

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
            logging.debug(f"Old: {self.current_controlnet}, New: {target_controlnet}")
            logging.debug("Cached controlnet not fould, loading new one")

            # Cleanup old controlnet
            self.controlnet = None
            self.memory_cleanup()

            if target_controlnet == ControlNetMode.NONE:
                self.current_controlnet = target_controlnet
                return

            # Load new controlnet if needed
            cn = ControlNetModel.from_pretrained(
                target_controlnet.value,
                resume_download=True,
                torch_dtype=torch.float32 if config.api.use_fp32 else torch.float16,
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
        else:
            logger.debug("No change in controlnet mode")

        # Clean memory
        self.memory_cleanup()

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

        generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        if job.data.scheduler:
            change_scheduler(
                model=pipe,
                scheduler=job.data.scheduler,
            )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            output_type = "pil"

            if "highres_fix" in job.flags:
                output_type = "latent"

            data = pipe(
                prompt=job.data.prompt,
                height=job.data.height,
                width=job.data.width,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                generator=generator,
                callback=txt2img_callback,
                num_images_per_prompt=job.data.batch_size,
            )

            if output_type == "latent":
                latents = data[0]  # type: ignore
                assert isinstance(latents, (torch.Tensor, torch.FloatTensor))

                flag = job.flags["highres_fix"]
                flag = HighResFixFlag.from_dict(flag)

                latents = scale_latents(
                    latents=latents,
                    scale=flag.scale,
                    latent_scale_mode=flag.latent_scale_mode,
                )

                self.memory_cleanup()

                data = pipe.img2img(
                    prompt=job.data.prompt,
                    image=latents,
                    num_inference_steps=flag.steps,
                    guidance_scale=job.data.guidance_scale,
                    negative_prompt=job.data.negative_prompt,
                    output_type="pil",
                    generator=generator,
                    callback=txt2img_callback,
                    strength=flag.strength,
                    return_dict=False,
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

        generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        # Preprocess the image
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
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

        generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        # Preprocess images
        input_image = convert_to_image(job.data.image).convert("RGB")
        input_image = resize(input_image, job.data.width, job.data.height)

        input_mask_image = convert_to_image(job.data.mask_image).convert("RGB")
        input_mask_image = ImageOps.invert(input_mask_image)
        input_mask_image = resize(input_mask_image, job.data.width, job.data.height)

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
                width=job.data.width,
                height=job.data.height,
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
                },
            )
        )

        return total_images

    def controlnet2img(self, job: ControlNetQueueEntry) -> List[Image.Image]:
        "Generate an image from an image and controlnet conditioning"

        if config.api.trace_model is True:
            raise ValueError(
                "ControlNet is not available with traced UNet, please disable tracing and reload the model."
            )

        logger.debug(f"Requested ControlNet: {job.data.controlnet}")
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

        generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        change_scheduler(model=pipe, scheduler=job.data.scheduler)

        # Preprocess the image
        from core.controlnet_preprocessing import image_to_controlnet_input

        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)
        input_image = image_to_controlnet_input(input_image, job.data)

        # Preprocess the prompt
        prompt_embeds, negative_embeds = get_weighted_text_embeddings(
            pipe=pipe,  # type: ignore # implements same protocol, but doesn't inherit
            prompt=job.data.prompt,
            uncond_prompt=job.data.negative_prompt,
        )

        total_images: List[Image.Image] = [input_image]

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                image=input_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
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
                    "image": convert_images_to_base64_grid(
                        total_images, quality=90, image_format="webp"
                    ),
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
        self.memory_cleanup()

        try:
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
        except Exception as e:
            self.memory_cleanup()
            raise e

        # Clean memory and return images
        self.memory_cleanup()
        return images

    def save(self, path: str = "converted", safetensors: bool = False):
        "Dump current pipeline to specified path"

        pipe = StableDiffusionPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
            safety_checker=self.safety_checker,
        )

        pipe.save_pretrained(path, safe_serialization=safetensors)

    def load_lora(
        self, lora: str, alpha_text_encoder: float = 0.5, alpha_unet: float = 0.5
    ):
        "Inject a LoRA model into the pipeline"

        logger.info(f"Loading LoRA model {lora} onto {self.model_id}...")

        if any(lora in l for l in self.loras):
            logger.info(f"LoRA model {lora} already loaded onto {self.model_id}")
            return

        if ".safetensors" in lora:
            load_safetensors_loras(
                self.text_encoder, self.unet, lora, alpha_text_encoder, alpha_unet
            )
        else:
            self.unet.load_attn_procs(
                pretrained_model_name_or_path_or_dict=lora,
                resume_download=True,
                use_auth_token=self.auth,
            )
        self.loras.append(lora)
        logger.info(f"LoRA model {lora} loaded successfully")
