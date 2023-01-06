from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import Txt2imgData


class Scheduler(Enum):
    euler = auto()
    euler_a = auto()
    ddim = auto()
    default = auto()


class SupportedModel(Enum):
    AnythingV3 = "Linaqruf/anything-v3.0"
    StableDiffusion1_5 = "runwayml/stable-diffusion-v1-5"

@dataclass
class Txt2ImgQueueEntry:
    job: "Txt2imgData"
    model: SupportedModel
    scheduler: Scheduler