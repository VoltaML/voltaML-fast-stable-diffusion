import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image, ImageOps
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from api.websockets.notification import Notification
from core.config import config
from core.flags import HighResFixFlag
from core.inference.base_model import InferenceModel
from core.inference.functions import convert_vaept_to_diffusers, load_pytorch_pipeline
from core.inference.pytorch.pipeline import StableDiffusionLongPromptWeightingPipeline
from core.inference.utilities import (
    change_scheduler,
    image_to_controlnet_input,
    scale_latents,
)
from core.inference_callbacks import (
    controlnet_callback,
    img2img_callback,
    inpaint_callback,
    txt2img_callback,
)
from core.types import (
    Backend,
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
        device: Union[str, torch.device] = "cuda",
        autoload: bool = True,
        bare: bool = False,
    ) -> None:
        super().__init__(model_id, device)

        self.backend: Backend = "PyTorch"
        self.bare: bool = bare

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
        self.controlnet: Optional[ControlNetModel] = None

        self.current_controlnet: str = ""

        self.vae_path: str = "default"
        self.unload_loras: List[str] = []
        self.unload_lycoris: List[str] = []
        self.textual_inversions: List[str] = []

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
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        self.feature_extractor = pipe.feature_extractor  # type: ignore
        self.requires_safety_checker = False  # type: ignore
        self.safety_checker = pipe.safety_checker  # type: ignore

        if not self.bare:
            # Autoload textual inversions
            for textural_inversion in config.api.autoloaded_textual_inversions:
                try:
                    self.load_textual_inversion(textural_inversion)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        f"Failed to load textual inversion {textural_inversion}: {e}"
                    )
                    websocket_manager.broadcast_sync(
                        Notification(
                            severity="error",
                            message=f"Failed to load textual inversion: {textural_inversion}",
                            title="Autoload Error",
                        )
                    )

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
            # Why the fuck do you think that's constant pylint?
            # Are you mentally insane?
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

        if hasattr(self, "original_vae"):
            del self.original_vae  # type: ignore

        if hasattr(self, "lora_injector"):
            from ..injectables import uninstall_lora_hook

            uninstall_lora_hook(self)

        self.memory_cleanup()

    def manage_optional_components(
        self,
        *,
        variations: bool = False,
        target_controlnet: str = "",
    ) -> None:
        "Cleanup old components"

        from ..injectables import load_lora_utilities

        load_lora_utilities(self)

        if not variations:
            self.image_encoder = None

        if self.current_controlnet != target_controlnet:
            logging.debug(f"Old: {self.current_controlnet}, New: {target_controlnet}")
            logging.debug("Cached controlnet not found, loading new one")

            # Cleanup old controlnet
            self.controlnet = None
            self.memory_cleanup()

            if not target_controlnet:
                self.current_controlnet = target_controlnet
                return

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

            cn.to(self.device)
            self.controlnet = cn
            self.current_controlnet = target_controlnet
        else:
            logger.debug("No change in controlnet mode")

        # Clean memory
        self.memory_cleanup()

    def create_pipe(
        self,
        controlnet: Optional[str] = "",
        seed: Optional[int] = -1,
        scheduler: Optional[Tuple[Any, bool]] = None,
    ) -> Tuple[StableDiffusionLongPromptWeightingPipeline, torch.Generator]:
        "Create a pipeline -- useful for reducing backend clutter."
        self.manage_optional_components(target_controlnet=controlnet or "")

        pipe = StableDiffusionLongPromptWeightingPipeline(
            parent=self,
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            controlnet=self.controlnet,
        )

        if config.api.device_type == "directml":
            generator = torch.Generator().manual_seed(seed)  # type: ignore
        else:
            generator = torch.Generator(config.api.device).manual_seed(seed)  # type: ignore

        if scheduler:
            change_scheduler(
                model=pipe,
                scheduler=scheduler[0],  # type: ignore
                use_karras_sigmas=scheduler[1],
            )

        return pipe, generator

    def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image from a prompt"

        from core.shared_dependent import progress

        task = progress.add_task("Text2Image", total=job.data.batch_count, completed=0)

        pipe, generator = self.create_pipe(
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

        total_images: List[Image.Image] = []

        for _ in range(job.data.batch_count):
            output_type = "pil"

            if "highres_fix" in job.flags:
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
                    self_attention_scale=job.data.self_attention_scale,
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
            progress.advance(task)

        progress.remove_task(task)

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

        from core.shared_dependent import progress

        task = progress.add_task("Image2Image", total=job.data.batch_count, completed=0)

        pipe, generator = self.create_pipe(
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
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
            progress.advance(task)

        progress.remove_task(task)

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

        from core.shared_dependent import progress

        task = progress.add_task("Inpaint", total=job.data.batch_count, completed=0)

        pipe, generator = self.create_pipe(
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

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
            progress.advance(task)

        progress.remove_task(task)

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

        from core.shared_dependent import progress

        task = progress.add_task("Controlnet", total=job.data.batch_count, completed=0)

        if config.api.trace_model is True:
            raise ValueError(
                "ControlNet is not available with traced UNet, please disable tracing and reload the model."
            )

        logger.debug(f"Requested ControlNet: {job.data.controlnet}")
        pipe, generator = self.create_pipe(
            controlnet=job.data.controlnet,
            seed=job.data.seed,
            scheduler=(job.data.scheduler, job.data.use_karras_sigmas),
        )

        # Preprocess the image
        logger.debug(f"Requested dim: W{job.data.width}xH{job.data.height}")

        input_image = convert_to_image(job.data.image)
        logger.debug(f"Input image size: {input_image.size}")
        input_image = resize(input_image, job.data.width, job.data.height)
        logger.debug(f"Resized image size: {input_image.size}")

        # Preprocess the image if needed
        if not job.data.is_preprocessed:
            input_image = image_to_controlnet_input(input_image, job.data)
            logger.debug(f"Preprocessed image size: {input_image.size}")

        total_images: List[Image.Image] = [input_image]

        for _ in range(job.data.batch_count):
            data = pipe(
                prompt=job.data.prompt,
                negative_prompt=job.data.negative_prompt,
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

            images = data[0]  # type: ignore
            assert isinstance(images, List)

            total_images.extend(images)  # type: ignore
            progress.advance(task)

        progress.remove_task(task)

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="controlnet",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_images_to_base64_grid(
                        total_images
                        if job.data.return_preprocessed
                        else total_images[1:],
                        quality=90,
                        image_format="webp",
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
            elif isinstance(job, ControlNetQueueEntry):
                images = self.controlnet2img(job)
            else:
                raise ValueError("Invalid job type for this pipeline")
        except Exception as e:
            self.memory_cleanup()
            raise e
        if len(self.unload_loras) != 0:
            for l in self.unload_loras:
                try:
                    self.lora_injector.remove_lora(l)  # type: ignore
                    logger.debug(f"Unloading LoRA: {l}")
                except KeyError:
                    pass
            self.unload_loras.clear()
        if len(self.unload_lycoris) != 0:  # type: ignore
            for l in self.unload_lycoris:  # type: ignore
                try:
                    self.lora_injector.remove_lycoris(l)  # type: ignore
                    logger.debug(f"Unloading LyCORIS: {l}")
                except KeyError:
                    pass

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

    def load_textual_inversion(self, textual_inversion: str):
        "Inject a textual inversion model into the pipeline"

        logger.info(
            f"Loading textual inversion model {textual_inversion} onto {self.model_id}..."
        )

        if any(textual_inversion in l for l in self.textual_inversions):
            logger.info(
                f"Textual inversion model {textual_inversion} already loaded onto {self.model_id}"
            )
            return

        pipe = StableDiffusionLongPromptWeightingPipeline(
            parent=self,
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            controlnet=self.controlnet,
        )

        token = Path(textual_inversion).stem
        logger.info(f"Loading token {token} for textual inversion model")

        pipe.load_textual_inversion(textual_inversion, token=token)

        self.textual_inversions.append(textual_inversion)
        logger.info(f"Textual inversion model {textual_inversion} loaded successfully")
        logger.debug(f"All added tokens: {self.tokenizer.added_tokens_encoder}")

    def tokenize(self, text: str):
        "Return the vocabulary of the tokenizer"

        return [i.replace("</w>", "") for i in self.tokenizer.tokenize(text=text)]
