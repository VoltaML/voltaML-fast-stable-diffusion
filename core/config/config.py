from dataclasses import dataclass, field

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import DIFFUSERS_CACHE


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    model_id: str = "andite/anything-v4.0"
    scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler
    )


@dataclass
class Img2ImgConfig:
    "Configuration for the image to image pipeline"

    model_id: str = "andite/anything-v4.0"
    scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler
    )


@dataclass
class Configuration:
    "Main configuration class for the application"

    txt2img: Txt2ImgConfig = field(default=Txt2ImgConfig())
    cache_dir: str = field(default=DIFFUSERS_CACHE)
