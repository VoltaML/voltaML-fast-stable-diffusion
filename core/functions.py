import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.utils import convert_image_to_base64

logger = logging.getLogger(__name__)


def cheap_approximation(sample: torch.Tensor):
    # Credit to Automatic111 stable-diffusion-webui
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    coefs = torch.tensor(
        [
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473],
        ]
    ).to(sample.device)

    x_sample = torch.einsum("lxy,lr -> rxy", sample, coefs)

    return x_sample


def pytorch_callback(data: dict):
    "Send a websocket message to the client with the progress percentage and partial image"

    _x: torch.Tensor = data["x"][0]
    step = int(data["i"]) + 1
    send_image = step % shared.image_decode_steps == 0
    image: str = ""

    if send_image:
        decoded_rgb = cheap_approximation(_x)
        decoded_rgb = torch.clamp((decoded_rgb + 1.0) / 2.0, min=0.0, max=1.0)
        decoded_rgb = 255.0 * np.moveaxis(decoded_rgb.cpu().numpy(), 0, 2)
        decoded_rgb = decoded_rgb.astype(np.uint8)
        image = convert_image_to_base64(Image.fromarray(decoded_rgb))

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="txt2img",
            data={
                "progress": int((step / shared.current_steps) * 100),
                "current_step": step,
                "total_steps": shared.current_steps,
                "image": image,
            },
        )
    )
