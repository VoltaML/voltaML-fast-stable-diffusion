import asyncio
import logging
from pathlib import Path
from typing import Any, List

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from PIL import Image, ImageOps
from transformers.models.clip.modeling_clip import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.flags import RefinerFlag
from core.inference.base_model import InferenceModel
from core.inference.functions import convert_vaept_to_diffusers, load_pytorch_pipeline
from core.inference.pytorch.lwp import get_weighted_text_embeddings
from core.inference.pytorch.lwp_sdxl import StableDiffusionXLLongPromptWeightingPipeline
from core.inference_callbacks import (
    img2img_callback,
    inpaint_callback,
    txt2img_callback,
)
from core.schedulers import change_scheduler
from core.types import (
    Backend,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

logger = logging.getLogger(__name__)


class SDXLStableDiffusion(InferenceModel):
    "High level model wrapper for SDXL models"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        autoload: bool = True,
        bare: bool = False,
    ) -> None:
        super().__init__(model_id, device)

        self.backend: Backend = "SDXL"
        self.bare: bool = bare

        # Components
        self.vae: AutoencoderKL
        self.unet: UNet2DConditionModel
        self.text_encoder: CLIPTextModel
        self.text_encoder_2: CLIPTextModelWithProjection
        self.tokenizer: CLIPTokenizer
        self.tokenizer_2: CLIPTokenizer
        self.force_zeros: bool
        self.aesthetic_score: bool
        self.scheduler: Any
        self.final_offload_hook: Any = None
        self.image_encoder: Any

        self.vae_path: str = "default"

        if autoload:
            self.load()

    def load(self):
        "Load the model from HuggingFace"

        logger.info(f"Loading {self.model_id} with {config.api.data_type}")

        pipe = load_pytorch_pipeline(
            self.model_id,
            device=self.device,
            optimize=not self.bare,
        )

        self.vae = pipe.vae  # type: ignore
        self.unet = pipe.unet  # type: ignore
        self.text_encoder = pipe.text_encoder  # type: ignore
        self.text_encoder_2 = pipe.text_encoder_2  # type: ignore
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.tokenizer_2 = pipe.tokenizer_2  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        if hasattr(pipe.config, "requires_aesthetics_score"):
            self.aesthetic_score = pipe.config.requires_aesthetics_score  # type: ignore
        else:
            self.aesthetic_score = False
        self.force_zeros = pipe.config.force_zeros_for_empty_prompt  # type: ignore
        if hasattr(pipe, "final_offload_hook"):
            self.final_offload_hook = pipe.final_offload_hook

        # Free up memory
        del pipe
        self.memory_cleanup()

    def change_vae(self, vae: str) -> None:
        "Change the vae to the one specified"

        if self.vae_path == "default":
            setattr(self, "original_vae", self.vae)

        old_vae = getattr(self, "original_vae")
        if vae == "default":
            self.vae = old_vae
        else:
            if Path(vae).is_dir():
                self.vae = AutoencoderKL.from_pretrained(vae)  # type: ignore
            else:
                self.vae = convert_vaept_to_diffusers(vae).to(
                    device=old_vae.device, dtype=old_vae.dtype
                )
        # This is at the end 'cause I've read horror stories about pythons prefetch system
        self.vae_path = vae

    def unload(self) -> None:
        "Unload the model from memory"

        del (
            self.vae,
            self.unet,
            self.text_encoder,
            self.text_encoder_2,
            self.tokenizer,
            self.tokenizer_2,
            self.scheduler,
            self.aesthetic_score,
            self.force_zeros,
        )

        if hasattr(self, "original_vae"):
            del self.original_vae  # type: ignore

        self.memory_cleanup()

    def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image from a prompt"

        total_images: List[Image.Image] = []

        if config.api.device_type == "directml":
            generator = torch.Generator().manual_seed(job.data.seed)
        else:
            generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        pipe = self.create_pipe()

        if job.data.scheduler:
            change_scheduler(
                model=pipe,
                scheduler=job.data.scheduler,
                use_karras_sigmas=job.data.use_karras_sigmas,
            )

        for _ in range(job.data.batch_count):
            output_type = "pil"

            if "refiner" in job.flags:
                output_type = "latent"

            data = pipe.text2img(
                prompt=job.data.prompt,
                height=job.data.height,
                width=job.data.width,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                self_attention_scale=job.data.self_attention_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                generator=generator,
                callback=txt2img_callback,
                num_images_per_prompt=job.data.batch_size,
                return_dict=False,
            )

            if output_type == "latent":
                latents: torch.FloatTensor = data[0]  # type: ignore
                flags = RefinerFlag.from_dict(job.flags["refiner"])

                from core.shared_dependent import gpu

                unload = False
                if flags.model not in gpu.loaded_models:
                    asyncio.run(gpu.load_model(flags.model, "SDXL"))
                    unload = True
                model: SDXLStableDiffusion = gpu.loaded_models[flags.model]  # type: ignore
                if config.api.clear_memory_policy == "always":
                    self.memory_cleanup()
                pipe = model.create_pipe()
                data = pipe(
                    image=latents,
                    prompt=job.data.prompt,
                    height=job.data.height,
                    width=job.data.width,
                    strength=flags.strength,
                    num_inference_steps=flags.steps,
                    guidance_scale=job.data.guidance_scale,
                    self_attention_scale=job.data.self_attention_scale,
                    negative_prompt=job.data.negative_prompt,
                    generator=generator,
                    callback=txt2img_callback,
                    num_images_per_prompt=job.data.batch_size,
                    return_dict=False,
                    output_type="pil",
                )
                del model
                if unload:
                    asyncio.run(gpu.unload(flags.model))
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

    def create_pipe(self) -> StableDiffusionXLLongPromptWeightingPipeline:
        "Create an LWP-XL pipeline"

        return StableDiffusionXLLongPromptWeightingPipeline(
            parent=self,
            vae=self.vae,
            unet=self.unet,  # type: ignore
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            scheduler=self.scheduler,
            force_zeros=self.force_zeros,
            aesthetic_score=self.aesthetic_score,
            final_offload_hook=self.final_offload_hook,
        )

    def img2img(self, job: Img2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image from an image"

        pipe = self.create_pipe()

        if config.api.device_type == "directml":
            generator = torch.Generator().manual_seed(job.data.seed)
        else:
            generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        change_scheduler(
            model=pipe,
            scheduler=job.data.scheduler,
            use_karras_sigmas=job.data.use_karras_sigmas,
        )

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
                self_attention_scale=job.data.self_attention_scale,
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

        pipe = self.create_pipe()

        if config.api.device_type == "directml":
            generator = torch.Generator().manual_seed(job.data.seed)
        else:
            generator = torch.Generator(config.api.device).manual_seed(job.data.seed)

        change_scheduler(
            model=pipe,
            scheduler=job.data.scheduler,
            use_karras_sigmas=job.data.use_karras_sigmas,
        )

        # Preprocess images
        input_image = convert_to_image(job.data.image).convert("RGB")
        input_image = resize(input_image, job.data.width, job.data.height)

        input_mask_image = convert_to_image(job.data.mask_image).convert("RGB")
        input_mask_image = ImageOps.invert(input_mask_image)
        input_mask_image = resize(input_mask_image, job.data.width, job.data.height)

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            if isinstance(pipe, StableDiffusionInpaintPipeline):
                (
                    prompt_embeds,
                    _,
                    negative_prompt_embeds,
                    _,
                ) = get_weighted_text_embeddings(
                    pipe=self,  # type: ignore
                    prompt=job.data.prompt,
                    uncond_prompt=job.data.negative_prompt,
                )

                data = pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,  # type: ignore
                    image=input_image,
                    mask_image=input_mask_image,
                    num_inference_steps=job.data.steps,
                    guidance_scale=job.data.guidance_scale,
                    negative_prompt_embeds=negative_prompt_embeds,  # type: ignore
                    output_type="pil",
                    generator=generator,
                    callback=inpaint_callback,
                    return_dict=False,
                    num_images_per_prompt=job.data.batch_size,
                    width=job.data.width,
                    height=job.data.height,
                )
            else:
                data = pipe.inpaint(
                    prompt=job.data.prompt,
                    image=input_image,
                    mask_image=input_mask_image,
                    num_inference_steps=job.data.steps,
                    guidance_scale=job.data.guidance_scale,
                    self_attention_scale=job.data.self_attention_scale,
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

    def generate(self, job: Job):
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

        pipe = StableDiffusionXLPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder_2=self.text_encoder_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            scheduler=self.scheduler,
        )

        pipe.save_pretrained(path, safe_serialization=safetensors)

    def load_textual_inversion(self, textual_inversion: str):
        "Inject a textual inversion model into the pipeline"

    def tokenize(self, text: str):
        "Return the vocabulary of the tokenizer"

        return [i.replace("</w>", "") for i in self.tokenizer.tokenize(text=text)]
