from dataclasses import dataclass, field

from diffusers.utils import DIFFUSERS_CACHE

from ..schedulers import Scheduler


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    model_id: str = "EleutherAI/gpt-neo-125M"
    scheduler: Scheduler = Scheduler.euler_a


@dataclass
class Img2ImgConfig:
    "Configuration for the image to image pipeline"

    model_id: str = "EleutherAI/gpt-neo-125M"
    scheduler: Scheduler = Scheduler.euler_a


@dataclass
class Configuration:
    "Main configuration class for the application"

    txt2img: Txt2ImgConfig = field(default=Txt2ImgConfig())
    cache_dir: str = field(default=DIFFUSERS_CACHE)
