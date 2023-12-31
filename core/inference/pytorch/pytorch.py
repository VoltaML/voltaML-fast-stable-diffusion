import importlib
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import requests
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from api.websockets.notification import Notification
from core.config import config
from core.flags import AnimateDiffFlag, DeepshrinkFlag, ScalecrafterFlag
from core.inference.base_model import InferenceModel
from core.inference.functions import (
    convert_vaept_to_diffusers,
    get_output_type,
    load_pytorch_pipeline,
)
from core.inference.pytorch.pipeline import StableDiffusionLongPromptWeightingPipeline
from core.inference.utilities import (
    change_scheduler,
    create_generator,
    image_to_controlnet_input,
)
from core.inference_callbacks import callback
from core.optimizations import optimize_vae
from core.types import (
    Backend,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Job,
    SigmaScheduler,
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

        if config.api.autoloaded_vae.get(self.model_id):
            try:
                self.change_vae(config.api.autoloaded_vae[self.model_id])
            except FileNotFoundError as e:
                logger.error(f"Failed to load autoloaded VAE: {e}")

        if not self.bare:
            # Autoload textual inversions
            for textural_inversion in config.api.autoloaded_textual_inversions:
                try:
                    self.load_textual_inversion(textural_inversion)
                except Exception as e:
                    logger.warning(
                        f"({e.__class__.__name__}) Failed to load textual inversion {textural_inversion}: {e}"
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

        logger.info(f"Changing VAE to {vae}")

        if self.vae_path == "default":
            setattr(self, "original_vae", self.vae)

        old_vae = getattr(self, "original_vae")
        # Not sure what I needed this for, but whatever
        dtype = config.api.load_dtype
        device = self.unet.device

        if hasattr(self.text_encoder, "v_offload_device"):
            device = torch.device("cpu")

        if vae == "default":
            self.vae = old_vae
        else:
            if len(vae.split("/")) == 2:
                cont = requests.get(
                    f"https://huggingface.co/{vae}/raw/main/config.json"
                ).json()["_class_name"]
                cont = getattr(importlib.import_module("diffusers"), cont)
                self.vae = cont.from_pretrained(vae).to(device, dtype)
                if not hasattr(self.vae.config, "block_out_channels"):
                    setattr(
                        self.vae.config,
                        "block_out_channels",
                        (
                            1,
                            1,
                            1,
                            1,
                        ),
                    )
            else:
                if Path(vae).exists():
                    if Path(vae).is_dir():
                        self.vae = ModelMixin.from_pretrained(vae)  # type: ignore
                    else:
                        self.vae = convert_vaept_to_diffusers(vae).to(device, dtype)
                else:
                    raise FileNotFoundError(f"{vae} is not a valid path")

        # Check if text_encoder has v_offload_device, because it always
        # gets wholly offloaded instead of being sequentially offloaded
        if hasattr(self.text_encoder, "v_offload_device"):
            from core.optimizations.offload import set_offload

            self.vae = set_offload(self.vae, torch.device(config.api.device))  # type: ignore
        self.vae = optimize_vae(self.vae)  # type: ignore
        self.vae_path = vae

    def unload(self) -> None:
        "Unload the model from memory"

        del (
            self.vae,
            self.unet,
            self.text_encoder,
            self.tokenizer,
            self.scheduler,
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
            self.image_encoder = None  # type: ignore

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
                torch_dtype=config.api.load_dtype,
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
        else:
            logger.debug("No change in controlnet mode")

    def create_pipe(
        self,
        controlnet: Optional[str] = "",
        scheduler: Optional[Tuple[Any, SigmaScheduler]] = None,
        sampler_settings: Optional[dict] = None,
    ) -> StableDiffusionLongPromptWeightingPipeline:
        "Create a pipeline -- useful for reducing backend clutter."
        self.manage_optional_components(target_controlnet=controlnet or "")

        pipe = StableDiffusionLongPromptWeightingPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            controlnet=self.controlnet,
        )
        pipe.parent = self

        if scheduler:
            change_scheduler(
                model=pipe,
                scheduler=scheduler[0],  # type: ignore
                sigma_type=scheduler[1],
                sampler_settings=sampler_settings,
            )

        return pipe

    def txt2img(self, job: Txt2ImgQueueEntry) -> Union[List[Image.Image], torch.Tensor]:
        "Generate an image from a prompt"

        pipe = self.create_pipe(
            scheduler=(job.data.scheduler, job.data.sigmas),
            sampler_settings=job.data.sampler_settings,
        )

        generator = create_generator(job.data.seed)

        total_images: Union[List[Image.Image], torch.Tensor] = []
        output_type = get_output_type(job)

        deepshrink = None
        if "deepshrink" in job.flags:
            deepshrink = DeepshrinkFlag.from_dict(job.flags["deepshrink"])

        scalecrafter = None
        if "scalecrafter" in job.flags:
            scalecrafter = ScalecrafterFlag.from_dict(job.flags["scalecrafter"])

        animatediff = AnimateDiffFlag(
            motion_model="data/motion-models/v3_sd15_mm.ckpt",
        )
        if "animatediff" in job.flags:
            animatediff = AnimateDiffFlag.from_dict(job.flags["animatediff"])

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            if animatediff is not None and animatediff.use_pia:
                base_image = pipe(  # type: ignore
                    generator=generator,
                    prompt=job.data.prompt,
                    height=job.data.height,
                    width=job.data.width,
                    num_inference_steps=job.data.steps,
                    guidance_scale=job.data.guidance_scale,
                    self_attention_scale=job.data.self_attention_scale,
                    negative_prompt=job.data.negative_prompt,
                    output_type=output_type,
                    callback=callback,
                    num_images_per_prompt=job.data.batch_size,
                    seed=job.data.seed,
                    prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                    deepshrink=deepshrink,
                    scalecrafter=scalecrafter,
                    animatediff=None,
                )[0][0]
                data = pipe(
                    generator=generator,
                    prompt=job.data.prompt,
                    image=base_image,
                    height=job.data.height,
                    width=job.data.width,
                    num_inference_steps=job.data.steps,
                    guidance_scale=job.data.guidance_scale,
                    self_attention_scale=job.data.self_attention_scale,
                    negative_prompt=job.data.negative_prompt,
                    output_type=output_type,
                    callback=callback,
                    num_images_per_prompt=job.data.batch_size,
                    seed=job.data.seed,
                    prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                    deepshrink=deepshrink,
                    scalecrafter=scalecrafter,
                    animatediff=animatediff,
                )
            else:
                data = pipe(
                    generator=generator,
                    prompt=job.data.prompt,
                    height=job.data.height,
                    width=job.data.width,
                    num_inference_steps=job.data.steps,
                    guidance_scale=job.data.guidance_scale,
                    self_attention_scale=job.data.self_attention_scale,
                    negative_prompt=job.data.negative_prompt,
                    output_type=output_type,
                    callback=callback,
                    num_images_per_prompt=job.data.batch_size,
                    seed=job.data.seed,
                    prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                    deepshrink=deepshrink,
                    scalecrafter=scalecrafter,
                    animatediff=animatediff,
                )

            images: Union[List[Image.Image], torch.Tensor] = data[0]  # type: ignore

            if not isinstance(images, List):
                total_images = images
            else:
                assert isinstance(total_images, List)
                total_images.extend(images)

        if isinstance(total_images, List):
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="txt2img",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images,  # type: ignore
                            quality=config.api.image_quality,
                            image_format=config.api.image_extension,
                        ),
                    },
                )
            )

        return total_images

    def img2img(self, job: Img2ImgQueueEntry) -> Union[List[Image.Image], torch.Tensor]:
        "Generate an image from an image"

        pipe = self.create_pipe(
            scheduler=(job.data.scheduler, job.data.sigmas),
            sampler_settings=job.data.sampler_settings,
        )

        generator = create_generator(job.data.seed)

        # Preprocess the image
        if isinstance(job.data.image, (str, bytes, Image.Image)):
            input_image = convert_to_image(job.data.image)
            input_image = resize(input_image, job.data.width, job.data.height)
        else:
            input_image = job.data.image

        total_images: Union[List[Image.Image], torch.Tensor] = []
        output_type = get_output_type(job)

        deepshrink = None
        if "deepshrink" in job.flags:
            deepshrink = DeepshrinkFlag.from_dict(job.flags["deepshrink"])

        animatediff = AnimateDiffFlag(
            motion_model="data/motion-models/v3_sd15_mm.ckpt",
        )
        if "animatediff" in job.flags:
            animatediff = AnimateDiffFlag.from_dict(job.flags["animatediff"])

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            data = pipe(
                generator=generator,
                prompt=job.data.prompt,
                image=input_image,
                height=job.data.height,  # technically isn't needed, but it's here for consistency sake
                width=job.data.width,  # technically isn't needed, but it's here for consistency sake
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                self_attention_scale=job.data.self_attention_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                callback=callback,
                strength=job.data.strength,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                seed=job.data.seed,
                prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                deepshrink=deepshrink,
                animatediff=animatediff,
            )

            images: Union[List[Image.Image], torch.Tensor] = data[0]  # type: ignore

            if not isinstance(images, List):
                total_images = images
            else:
                assert isinstance(total_images, List)
                total_images.extend(images)

        if isinstance(total_images, List):
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="img2img",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images,  # type: ignore
                            quality=config.api.image_quality,
                            image_format=config.api.image_extension,
                        ),
                    },
                )
            )

        return total_images

    def inpaint(self, job: InpaintQueueEntry) -> Union[List[Image.Image], torch.Tensor]:
        "Generate an image from an image"

        pipe = self.create_pipe(
            scheduler=(job.data.scheduler, job.data.sigmas),
            sampler_settings=job.data.sampler_settings,
        )

        generator = create_generator(job.data.seed)

        # Preprocess images
        input_image = convert_to_image(job.data.image).convert("RGB")
        input_image = resize(input_image, job.data.width, job.data.height)

        input_mask_image = convert_to_image(job.data.mask_image).convert("RGB")
        input_mask_image = ImageOps.invert(input_mask_image)
        input_mask_image = resize(input_mask_image, job.data.width, job.data.height)

        total_images: Union[List[Image.Image], torch.Tensor] = []
        output_type = get_output_type(job)

        deepshrink = None
        if "deepshrink" in job.flags:
            deepshrink = DeepshrinkFlag.from_dict(job.flags["deepshrink"])

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            data = pipe(
                generator=generator,
                prompt=job.data.prompt,
                image=input_image,
                mask_image=input_mask_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                self_attention_scale=job.data.self_attention_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                callback=callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                width=job.data.width,
                height=job.data.height,
                seed=job.data.seed,
                prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                deepshrink=deepshrink,
            )

            images: Union[List[Image.Image], torch.Tensor] = data[0]  # type: ignore

            if not isinstance(images, List):
                total_images = images
            else:
                assert isinstance(total_images, List)
                total_images.extend(images)

        if isinstance(total_images, List):
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="inpainting",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images,  # type: ignore
                            quality=config.api.image_quality,
                            image_format=config.api.image_extension,
                        ),
                    },
                )
            )

        return total_images

    def controlnet2img(
        self, job: ControlNetQueueEntry
    ) -> Union[List[Image.Image], torch.Tensor]:
        "Generate an image from an image and controlnet conditioning"

        if config.api.trace_model is True:
            raise ValueError(
                "ControlNet is not available with traced UNet, please disable tracing and reload the model."
            )

        logger.debug(f"Requested ControlNet: {job.data.controlnet}")
        pipe = self.create_pipe(
            controlnet=job.data.controlnet,
            scheduler=(job.data.scheduler, job.data.sigmas),
            sampler_settings=job.data.sampler_settings,
        )

        generator = create_generator(job.data.seed)

        # Preprocess the image
        input_image = convert_to_image(job.data.image)
        input_image = resize(input_image, job.data.width, job.data.height)

        # Preprocess the image if needed
        if not job.data.is_preprocessed:
            input_image = image_to_controlnet_input(input_image, job.data)
            logger.debug(f"Preprocessed image size: {input_image.size}")

        total_images: Union[List[Image.Image], torch.Tensor] = []
        output_type = get_output_type(job)

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            data = pipe(
                generator=generator,
                prompt=job.data.prompt,
                negative_prompt=job.data.negative_prompt,
                image=input_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                output_type=output_type,
                callback=callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                controlnet_conditioning_scale=job.data.controlnet_conditioning_scale,
                height=job.data.height,
                width=job.data.width,
                seed=job.data.seed,
                prompt_expansion_settings=job.data.prompt_to_prompt_settings,
            )

            images: Union[List[Image.Image], torch.Tensor] = data[0]  # type: ignore

            if not isinstance(images, List):
                total_images = images
            else:
                assert isinstance(total_images, List)
                total_images.extend(images)

        if isinstance(total_images, List):
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="controlnet",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images  # type: ignore
                            if job.data.return_preprocessed
                            else total_images[1:],
                            quality=config.api.image_quality,
                            image_format=config.api.image_extension,
                        ),
                    },
                )
            )

        return total_images

    def generate(self, job: Job) -> Union[List[Image.Image], torch.Tensor]:
        "Generate images from the queue"

        logging.info(f"Adding job {job.data.id} to queue")

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
            for lora in self.unload_loras:
                try:
                    self.lora_injector.remove_lora(lora)  # type: ignore
                    logger.debug(f"Unloading LoRA: {lora}")
                except KeyError:
                    pass
            self.unload_loras.clear()
        if len(self.unload_lycoris) != 0:  # type: ignore
            for lora in self.unload_lycoris:  # type: ignore
                try:
                    self.lora_injector.remove_lycoris(lora)  # type: ignore
                    logger.debug(f"Unloading LyCORIS: {lora}")
                except KeyError:
                    pass

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

        if any(textual_inversion in lora for lora in self.textual_inversions):
            logger.info(
                f"Textual inversion model {textual_inversion} already loaded onto {self.model_id}"
            )
            return

        pipe = StableDiffusionLongPromptWeightingPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            controlnet=self.controlnet,
        )
        pipe.parent = self

        token = Path(textual_inversion).stem
        logger.info(f"Loading token {token} for textual inversion model")

        try:
            pipe.load_textual_inversion(textual_inversion, token=token)
        except ValueError as e:
            if "Loaded state dictonary is incorrect" in str(e):
                logger.info(f"Assuming {textual_inversion} is for SDXL, skipping")
                return
            raise e

        self.textual_inversions.append(textual_inversion)
        logger.info(f"Textual inversion model {textual_inversion} loaded successfully")
        logger.debug(f"All added tokens: {self.tokenizer.added_tokens_encoder}")

    def tokenize(self, text: str):
        "Return the vocabulary of the tokenizer"

        return [i.replace("</w>", "") for i in self.tokenizer.tokenize(text=text)]
