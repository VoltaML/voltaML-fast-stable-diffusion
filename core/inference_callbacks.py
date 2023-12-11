import time
from typing import Union, List
from io import BytesIO

import torch
from PIL import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.config import config
from core.errors import InferenceInterruptedError
from core.inference.utilities import cheap_approximation, numpy_to_pil, taesd
from core.utils import convert_images_to_base64_grid

last_image_time = time.time()


def callback(step: int, _timestep: int, tensor: torch.Tensor):
    "Callback for all processes that have steps and partial images."

    global last_image_time

    if shared.interrupt:
        shared.interrupt = False
        raise InferenceInterruptedError

    shared.current_done_steps += 1
    if step > shared.current_steps:
        shared.current_steps = shared.current_done_steps
    send_image = config.api.live_preview_method != "disabled" and (
        (time.time() - last_image_time > config.api.live_preview_delay)
    )

    images: List[Union[BytesIO, Image.Image]] = []
    if send_image:
        last_image_time = time.time()
        if config.api.live_preview_method == "approximation":
            for t in range(tensor.shape[0]):
                images.append(cheap_approximation(tensor[t]))
        else:
            for img in numpy_to_pil(taesd(tensor)):  # type: ignore
                images.append(img)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type=shared.current_method,  # type: ignore
            data={
                "progress": int(
                    (shared.current_done_steps / shared.current_steps) * 100
                ),
                "current_step": shared.current_done_steps,
                "total_steps": shared.current_steps,
                "image": convert_images_to_base64_grid(
                    images, quality=60, image_format="webp"
                )
                if send_image
                else "",
            },
        )
    )
