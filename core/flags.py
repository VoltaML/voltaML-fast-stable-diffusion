from dataclasses import dataclass
from typing import Literal

from dataclasses_json.api import DataClassJsonMixin

LatentScaleModel = Literal[
    "nearest",
    "area",
    "bilinear",
    "bislerp-original",
    "bislerp-tortured",
    "bicubic",
    "nearest-exact",
]


@dataclass
class Flag:
    "Base class for all flags"


@dataclass
class HighResFixFlag(Flag, DataClassJsonMixin):
    "Flag to fix high resolution images"

    scale: float = 2
    mode: Literal["latent", "image"] = "latent"

    # Image Upscaling
    image_upscaler: str = "RealESRGAN_x4plus_anime_6B"

    # Latent Upscaling
    latent_scale_mode: LatentScaleModel = "bislerp-tortured"
    antialiased: bool = False

    # Img2img
    strength: float = 0.7
    steps: int = 50
