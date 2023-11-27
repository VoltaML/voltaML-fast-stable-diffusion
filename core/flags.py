from dataclasses import dataclass, field
from typing import Literal

from dataclasses_json.api import DataClassJsonMixin

LatentScaleModel = Literal[
    "nearest",
    "area",
    "bilinear",
    "bislerp",
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
    latent_scale_mode: LatentScaleModel = "bislerp"
    antialiased: bool = False

    # Img2img
    strength: float = 0.7
    steps: int = 50
    antialiased: bool = False


@dataclass
class XLOriginalSize:
    width: int = 1024
    height: int = 1024


@dataclass
class SDXLFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL settings"

    original_size: XLOriginalSize = field(default_factory=XLOriginalSize)


@dataclass
class SDXLRefinerFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL refiners"

    steps: int = 50
    strength: float = 0.3
    model: str = ""
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
