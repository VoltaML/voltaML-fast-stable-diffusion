from dataclasses import dataclass, field
from typing import List, Literal

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
class XLFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL settings"

    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    original_size: List[int] = field(default_factory=lambda: [1024, 1024])


@dataclass
class XLRefinerFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL refiners"

    steps: int = 50
    strength: float = 0.3
    model: str = ""
