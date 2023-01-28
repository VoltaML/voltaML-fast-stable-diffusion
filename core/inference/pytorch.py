import gc
import logging
import os
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image

from api import websocket_manager
from api.websockets import Data
from core.config import config
from core.diffusers.kdiffusion import StableDiffusionKDiffusionPipeline
from core.schedulers import change_scheduler
from core.types import Img2ImgQueueEntry, KDiffusionScheduler, Txt2ImgQueueEntry
from core.utils import convert_image_to_base64, process_image

logger = logging.getLogger(__name__)


class PyTorchInferenceModel:
    "High level model wrapper for PyTorch models"

    def __init__(
        self,
        model_id: str,
        scheduler: KDiffusionScheduler = KDiffusionScheduler.euler_a,
        auth_token: str = os.environ["HUGGINGFACE_TOKEN"],
        use_f32: bool = False,
        device: str = "cuda",
        callback: Optional[Callable[[Dict], None]] = None,
        callback_steps: int = 10,
    ) -> None:
        self.use_f32: bool = use_f32
        self.auth: str = auth_token
        self.model_id: str = model_id
        self.device: str = device
        self.callback: Optional[Callable[[Dict], None]] = callback
        self.callback_steps: int = callback_steps
        self.model: Optional[StableDiffusionKDiffusionPipeline] = self.load()
        change_scheduler(model=self.model, scheduler=scheduler)

    def load(self) -> StableDiffusionKDiffusionPipeline:
        "Load the model from HuggingFace"

        logger.info(f"Loading {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.float32 if self.use_f32 else torch.float16,
            use_auth_token=self.auth,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            cache_dir=config.cache_dir,
        )

        logger.debug(f"Loaded {self.model_id} with {'f32' if self.use_f32 else 'f16'}")

        assert isinstance(pipe, StableDiffusionKDiffusionPipeline)
        pipe = pipe.to(self.device)
        pipe.set_scheduler("sample_euler_ancestral")
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

        if self.model is None:
            raise ValueError("Model not loaded")

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=self.model, scheduler=job.scheduler)

        data = self.model.txt2img(
            prompt=job.data.prompt,
            height=job.data.height,
            width=job.data.width,
            num_inference_steps=job.data.steps,
            guidance_scale=job.data.guidance_scale,
            negative_prompt=job.data.negative_prompt,
            output_type="pil",
            generator=generator,
            seed=job.data.seed,
            return_dict=False,
            callback=self.callback,
            use_karras_sigmas=job.use_karras_sigmas,
        )
        images: list[Image.Image] = data[0]

        websocket_manager.broadcast_sync(
            data=Data(
                data_type="txt2img",
                data={
                    "progress": 0,
                    "current_step": 0,
                    "total_steps": 0,
                    "image": convert_image_to_base64(images[0]),
                },
            )
        )

        return images

    def img2img(self, job: Img2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image from an image"

        if self.model is None:
            raise ValueError("Model not loaded")

        generator = torch.Generator("cuda").manual_seed(job.data.seed)

        change_scheduler(model=self.model, scheduler=job.scheduler)

        data = self.model.img2img(
            prompt=job.data.prompt,
            init_image=process_image(job.data.image),
            num_inference_steps=job.data.steps,
            guidance_scale=job.data.guidance_scale,
            negative_prompt=job.data.negative_prompt,
            output_type="pil",
            generator=generator,
            seed=job.data.seed,
            return_dict=False,
            callback=self.callback,
            use_karras_sigmas=job.use_karras_sigmas,
        )

        images: list[Image.Image] = data[0]

        return images

    def optimize(self) -> None:
        "Optimize the model for inference"

        logger.info("Optimizing model")

        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            self.model.enable_xformers_memory_efficient_attention()
            logging.info("Optimization: Enabled xformers memory efficient attention")
        except ModuleNotFoundError:
            logging.info(
                "Optimization: xformers not available, enabling attention slicing instead"
            )
            self.model.enable_attention_slicing()
            logging.info("Optimization: Enabled attention slicing")

        try:
            self.model.enable_traced_unet(self.model_id)
            logging.info("Optimization: Enabled traced UNet")
        except ValueError:
            logging.info("Optimization: Traced UNet not available")

        logger.info("Optimization complete")
