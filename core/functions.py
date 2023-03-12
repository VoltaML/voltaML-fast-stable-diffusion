import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.config import config
from core.errors import InferenceInterruptedError
from core.utils import convert_images_to_base64_grid

logger = logging.getLogger(__name__)

last_image_time = time.time()


def cheap_approximation(sample: torch.Tensor):
    "Convert a tensor of latents to RGB"

    # Credit to Automatic111 stable-diffusion-webui
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    coefs = (
        torch.tensor(
            [
                [0.298, 0.207, 0.208],
                [0.187, 0.286, 0.173],
                [-0.158, 0.189, 0.264],
                [-0.184, -0.271, -0.473],
            ]
        )
        .to(torch.float32)
        .to("cpu")
    )

    cast_sample = sample.to(torch.float32).to("cpu")

    x_sample = torch.einsum("lxy,lr -> rxy", cast_sample, coefs)

    return x_sample


def txt2img_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for txt2img with progress and partial image"

    images, send_image = pytorch_callback(step, _timestep, tensor)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="txt2img",
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(images) if send_image else "",
            },
        )
    )


def img2img_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for img2img with progress and partial image"

    images, send_image = pytorch_callback(step, _timestep, tensor)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="img2img",
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(images) if send_image else "",
            },
        )
    )


def inpaint_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for inpaint with progress and partial image"

    images, send_image = pytorch_callback(step, _timestep, tensor)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="inpainting",
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(images) if send_image else "",
            },
        )
    )


def image_variations_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for image variations with progress and partial image"

    images, send_image = pytorch_callback(step, _timestep, tensor)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="image_variations",
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(images) if send_image else "",
            },
        )
    )


def controlnet_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for controlnet with progress and partial image"

    images, send_image = pytorch_callback(step, _timestep, tensor)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="controlnet",
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(images) if send_image else "",
            },
        )
    )


def pytorch_callback(
    _step: int, _timestep: int, tensor: torch.Tensor
) -> Tuple[List[Image.Image], bool]:
    "Send a websocket message to the client with the progress percentage and partial image"

    global last_image_time  # pylint: disable=global-statement

    if shared.interrupt:
        shared.interrupt = False
        raise InferenceInterruptedError

    shared.current_done_steps += 1
    send_image: bool = time.time() - last_image_time > 2
    images: List[Image.Image] = []

    if send_image:
        last_image_time = time.time()
        for i in range(tensor.shape[0]):
            decoded_rgb = cheap_approximation(tensor[i])
            decoded_rgb = torch.clamp((decoded_rgb + 1.0) / 2.0, min=0.0, max=1.0)
            decoded_rgb = 255.0 * np.moveaxis(decoded_rgb.cpu().numpy(), 0, 2)
            decoded_rgb = decoded_rgb.astype(np.uint8)
            images.append(Image.fromarray(decoded_rgb))

    return images, send_image


def optimize_model(pipe: StableDiffusionPipeline) -> None:
    "Optimize the model for inference"

    logger.info("Optimizing model")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Optimization: Enabled xformers memory efficient attention")
    except ModuleNotFoundError:
        logger.info(
            "Optimization: xformers not available, enabling attention slicing instead"
        )
        pipe.enable_attention_slicing()
        logger.info("Optimization: Enabled attention slicing")

    if config.low_vram:
        pipe.enable_model_cpu_offload()
        logger.info("Optimization: Enabled model CPU offload")

    pipe.enable_vae_slicing()
    logger.info("Optimization: Enabled VAE slicing")

    # pipe.enable_vae_tiling()
    # logger.info("Optimization: Enabled VAE tiling")

    logger.info("Optimization complete")


def image_meta_from_file(path: Path) -> Dict[str, str]:
    "Return image metadata from a file"

    with path.open("rb") as f:
        image = Image.open(f)
        text = image.text  # type: ignore

        return text


def preprocess_image(image):
    "Preprocess an image for the model"

    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]  # type: ignore
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0  # type: ignore
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)  # type: ignore
    return image
