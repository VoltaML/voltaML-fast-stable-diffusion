import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from PIL import Image, ImageOps
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.clip.modeling_clip import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from api import websocket_manager
from api.websockets import Data
from api.websockets.notification import Notification
from core.config import config
from core.files import get_full_model_path
from core.flags import SDXLFlag, SDXLRefinerFlag, DeepshrinkFlag, ScalecrafterFlag
from core.inference.base_model import InferenceModel
from core.inference.functions import (
    convert_vaept_to_diffusers,
    get_output_type,
    load_pytorch_pipeline,
)
from core.inference.utilities import change_scheduler, create_generator
from core.inference_callbacks import callback
from core.optimizations import optimize_vae
from core.types import (
    Backend,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Job,
    SigmaScheduler,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image, resize

from .pipeline import StableDiffusionXLLongPromptWeightingPipeline

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

        self.backend: Backend = "PyTorch"
        self.type = "SDXL"
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
        self.text_encoder_2 = pipe.text_encoder_2  # type: ignore
        self.tokenizer = pipe.tokenizer  # type: ignore
        self.tokenizer_2 = pipe.tokenizer_2  # type: ignore
        self.scheduler = pipe.scheduler  # type: ignore
        if hasattr(pipe.config, "requires_aesthetics_score"):
            self.aesthetic_score = pipe.config.requires_aesthetics_score  # type: ignore
        else:
            self.aesthetic_score = False
        self.force_zeros = pipe.config.force_zeros_for_empty_prompt  # type: ignore

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

        if self.vae_path == "default":
            setattr(self, "original_vae", self.vae)

        old_vae = getattr(self, "original_vae")
        dtype = config.api.load_dtype
        device = self.unet.device

        if hasattr(self.text_encoder, "v_offload_device"):
            device = torch.device("cpu")

        if vae == "default":
            self.vae = old_vae
        else:
            full_path = get_full_model_path(vae)
            if full_path.is_dir():
                self.vae = AutoencoderKL.from_pretrained(full_path).to(  # type: ignore
                    device=device, dtype=dtype
                )
            else:
                self.vae = convert_vaept_to_diffusers(full_path.as_posix()).to(
                    device=device, dtype=dtype
                )

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

    def load_refiner(self, refiner: SDXLRefinerFlag, job) -> Tuple[Any, Any]:
        from core.shared_dependent import gpu

        unload = False

        if refiner.model not in gpu.loaded_models:
            gpu.load_model(refiner.model, "PyTorch", "SDXL")
            unload = True
        model: SDXLStableDiffusion = gpu.loaded_models[refiner.model]  # type: ignore
        if config.api.clear_memory_policy == "always":
            self.memory_cleanup()
        pipe = model.create_pipe(
            scheduler=(job.data.scheduler, job.data.sigmas),
            sampler_settings=job.data.sampler_settings,
        )
        unl = lambda: ""
        if unload:

            def unll():
                nonlocal model, refiner

                del model
                gpu.unload(refiner.model)

            unl = unll

        return pipe, unl

    def create_pipe(
        self,
        controlnet: Optional[str] = "",
        scheduler: Optional[Tuple[Any, SigmaScheduler]] = None,
        sampler_settings: Optional[dict] = None,
    ) -> StableDiffusionXLLongPromptWeightingPipeline:
        "Create an LWP-XL pipeline"

        # self.manage_optional_components(target_controlnet=controlnet or "")

        pipe = StableDiffusionXLLongPromptWeightingPipeline(
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

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            xl_flag = None
            if "sdxl" in job.flags:
                xl_flag = SDXLFlag.from_dict(job.flags["sdxl"])

            refiner = None
            if "refiner" in job.flags:
                output_type = "latent"
                refiner = SDXLRefinerFlag.from_dict(job.flags["refiner"])

            refiner_model, unload = None, lambda: ""
            if config.api.sdxl_refiner == "joint" and refiner is not None:
                refiner_model, unload = self.load_refiner(refiner, job)

            original_size = None
            if xl_flag:
                original_size = [
                    xl_flag.original_size.height,
                    xl_flag.original_size.width,
                ]

            data = pipe(
                original_size=original_size,
                generator=generator,
                prompt=job.data.prompt,
                height=job.data.height,
                width=job.data.width,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                callback=callback,
                num_images_per_prompt=job.data.batch_size,
                seed=job.data.seed,
                self_attention_scale=job.data.self_attention_scale,
                prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                refiner=refiner,
                refiner_model=refiner_model,
                deepshrink=deepshrink,
                scalecrafter=scalecrafter,
            )

            if refiner is not None and config.api.sdxl_refiner == "separate":
                latents: torch.FloatTensor = data[0]  # type: ignore

                refiner_model, unload = self.load_refiner(refiner, job)

                data = pipe(
                    aesthetic_score=refiner.aesthetic_score,
                    negative_aesthetic_score=refiner.negative_aesthetic_score,
                    original_size=original_size,
                    image=latents,
                    generator=generator,
                    prompt=job.data.prompt,
                    height=job.data.height,
                    width=job.data.width,
                    strength=refiner.strength,
                    num_inference_steps=refiner.steps,
                    guidance_scale=job.data.guidance_scale,
                    negative_prompt=job.data.negative_prompt,
                    callback=callback,
                    num_images_per_prompt=job.data.batch_size,
                    return_dict=False,
                    output_type=output_type,
                    seed=job.data.seed,
                    self_attention_scale=job.data.self_attention_scale,
                    prompt_expansion_settings=job.data.prompt_to_prompt_settings,
                )

            images: Union[List[Image.Image], torch.Tensor] = data[0]  # type: ignore

            if not isinstance(images, List):
                total_images = images
            else:
                assert isinstance(total_images, List)
                total_images.extend(images)

            unload()

        if isinstance(total_images, List):
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

        xl_flag = None
        if "sdxl" in job.flags:
            xl_flag = SDXLFlag.from_dict(job.flags["sdxl"])

        original_size = None
        if xl_flag:
            original_size = [
                xl_flag.original_size.height,
                xl_flag.original_size.width,
            ]

        deepshrink = None
        if "deepshrink" in job.flags:
            deepshrink = DeepshrinkFlag.from_dict(job.flags["deepshrink"])

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            data = pipe(
                original_size=original_size,
                generator=generator,
                prompt=job.data.prompt,
                image=input_image,  # type: ignore
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                width=job.data.width,
                height=job.data.height,
                output_type=output_type,
                callback=callback,
                strength=job.data.strength,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                seed=job.data.seed,
                self_attention_scale=job.data.self_attention_scale,
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

        xl_flag = None
        if "sdxl" in job.flags:
            xl_flag = SDXLFlag.from_dict(job.flags["sdxl"])

        original_size = None
        if xl_flag:
            original_size = [
                xl_flag.original_size.height,
                xl_flag.original_size.width,
            ]

        deepshrink = None
        if "deepshrink" in job.flags:
            deepshrink = DeepshrinkFlag.from_dict(job.flags["deepshrink"])

        for _ in tqdm(range(job.data.batch_count), desc="Queue", position=1):
            data = pipe(
                original_size=original_size,
                generator=generator,
                prompt=job.data.prompt,
                image=input_image,
                mask_image=input_mask_image,
                num_inference_steps=job.data.steps,
                guidance_scale=job.data.guidance_scale,
                negative_prompt=job.data.negative_prompt,
                output_type=output_type,
                callback=callback,
                return_dict=False,
                num_images_per_prompt=job.data.batch_size,
                width=job.data.width,
                height=job.data.height,
                seed=job.data.seed,
                self_attention_scale=job.data.self_attention_scale,
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
                            total_images, quality=90, image_format="webp"
                        ),
                    },
                )
            )

        return total_images

    def generate(self, job: Job) -> Union[List[Image.Image], torch.Tensor]:
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

        logger.info(
            f"Loading textual inversion model {textual_inversion} onto {self.model_id}..."
        )

        if any(textual_inversion in lora for lora in self.textual_inversions):
            logger.info(
                f"Textual inversion model {textual_inversion} already loaded onto {self.model_id}"
            )
            return

        pipe = StableDiffusionXLPipeline(
            vae=self.vae,
            unet=self.unet,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            scheduler=self.scheduler,
        )
        pipe.parent = self

        token = Path(textual_inversion).stem
        logger.info(f"Loading token {token} for textual inversion model")

        state_dict = load_file(textual_inversion)

        try:
            pipe.load_textual_inversion(
                state_dict["clip_g"],  # type: ignore
                token=token,
                text_encoder=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer_2,
            )
            pipe.load_textual_inversion(
                state_dict["clip_l"],  # type: ignore
                token=token,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
            )
        except KeyError:
            logger.info(f"Assuming {textual_inversion} is for non SDXL model, skipping")
            return

        self.textual_inversions.append(textual_inversion)
        logger.info(f"Textual inversion model {textual_inversion} loaded successfully")
        logger.debug(f"All added tokens: {self.tokenizer.added_tokens_encoder}")

    def tokenize(self, text: str):
        "Return the vocabulary of the tokenizer"

        return [i.replace("</w>", "") for i in self.tokenizer.tokenize(text=text)]
