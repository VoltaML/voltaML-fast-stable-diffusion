import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.errors import InferenceInterruptedError
from core.types import ImageMetadata
from core.utils import convert_images_to_base64_grid

logger = logging.getLogger(__name__)


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
        .to(torch.float16)
        .to(sample.device)
    )

    cast_sample = sample.to(torch.float16).to(sample.device)

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
            data_type="inpaint",
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


def pytorch_callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Send a websocket message to the client with the progress percentage and partial image"

    if shared.interrupt:
        shared.interrupt = False
        raise InferenceInterruptedError

    shared.current_done_steps += 1
    send_image = step % shared.image_decode_steps == 0
    images: List[Image.Image] = []

    if send_image:
        for i in range(tensor.shape[0]):
            decoded_rgb = cheap_approximation(tensor[i])
            decoded_rgb = torch.clamp((decoded_rgb + 1.0) / 2.0, min=0.0, max=1.0)
            decoded_rgb = 255.0 * np.moveaxis(decoded_rgb.cpu().numpy(), 0, 2)
            decoded_rgb = decoded_rgb.astype(np.uint8)
            images.append(Image.fromarray(decoded_rgb))

    return images, send_image


def image_meta_from_file(path: Path) -> ImageMetadata:
    "Return image metadata from a file"

    with path.open("rb") as f:
        image = Image.open(f)
        text = image.text  # type: ignore

        try:
            metadata = ImageMetadata(
                prompt=text["prompt"],
                negative_prompt=text["negative_prompt"],
                height=int(text["height"]),
                width=int(text["width"]),
                seed=text["seed"],
                guidance_scale=float(text["guidance_scale"]),
                steps=int(text["steps"]),
                model=text["model"],
            )
        except KeyError:
            metadata = ImageMetadata(
                prompt="",
                negative_prompt="",
                height=0,
                width=0,
                seed="",
                guidance_scale=0.0,
                steps=0,
                model="",
            )

    return metadata


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
