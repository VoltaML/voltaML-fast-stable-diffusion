from typing import Callable, Optional

from diffusers import AutoencoderTiny
import torch
import numpy as np
from PIL import Image

from core import shared
from core.config import config

taesd_model = None


def taesd(
    samples: torch.Tensor, height: Optional[int] = None, width: Optional[int] = None
) -> torch.Tensor:
    global taesd_model  # pylint: disable=global-statement

    if taesd_model is None:
        model = "madebyollin/taesd"
        if False:  # TODO: if is_sdxl:
            model = "madebyollin/taesdxl"
        taesd_model = AutoencoderTiny.from_pretrained(
            model, torch_dtype=torch.float16
        ).to(  # type: ignore
            config.api.device
        )

    return decode_latents(
        lambda sample: taesd_model.decode(sample).sample,  # type: ignore
        samples.to(torch.float16),
        height=height or samples[0].shape[1] * 8,
        width=width or samples[0].shape[2] * 8,
    )


def cheap_approximation(sample: torch.Tensor) -> Image.Image:
    "Convert a tensor of latents to RGB"

    # Credit to Automatic111 stable-diffusion-webui
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    coeffs = [
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]
    if False:  # TODO: if is_sdxl:
        coeffs = [
            [0.3448, 0.4168, 0.4395],
            [-0.1953, -0.0290, 0.0250],
            [0.1074, 0.0886, -0.0163],
            [-0.3730, -0.2499, -0.2088],
        ]
    coeffs = torch.tensor(coeffs, dtype=torch.float32, device="cpu")

    decoded_rgb = torch.einsum(
        "lxy,lr -> rxy", sample.to(torch.float32).to("cpu"), coeffs
    )
    decoded_rgb = torch.clamp((decoded_rgb + 1.0) / 2.0, min=0.0, max=1.0)
    decoded_rgb = 255.0 * np.moveaxis(decoded_rgb.cpu().numpy(), 0, 2)
    decoded_rgb = decoded_rgb.astype(np.uint8)

    return Image.fromarray(decoded_rgb)


def full_vae(
    samples: torch.Tensor,
    overwrite: Callable[[torch.Tensor], torch.Tensor],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    return decode_latents(
        overwrite,
        samples,
        height or samples[0].shape[1] * 8,
        width or samples[0].shape[2] * 8,
    )


def decode_latents(
    decode_lambda: Callable[[torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    "Decode latents"
    latents = 1 / 0.18215 * latents
    image = decode_lambda(latents)  # type: ignore
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    img = image[:, :height, :width, :]
    return img


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
