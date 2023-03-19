import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.errors import InferenceInterruptedError
from core.utils import convert_images_to_base64_grid

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
