from dataclasses import dataclass
from typing import Literal

from dataclasses_json.api import DataClassJsonMixin

LatentScaleModel = Literal[
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "nearest-exact",
]


@dataclass
class Flag:
    "Base class for all flags"


@dataclass
class HighResFixFlag(Flag, DataClassJsonMixin):
    "Flag to fix high resolution images"

    scale: int
    latent_scale_mode: LatentScaleModel = "bilinear"
    strength: float = 0.7
    steps: int = 50
    antialiased: bool = False
