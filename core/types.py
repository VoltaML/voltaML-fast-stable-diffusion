from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal
from uuid import uuid4


class Scheduler(Enum):
    euler = auto()
    euler_a = auto()
    ddim = auto()
    default = auto()


@dataclass
class Txt2imgData:
    prompt: str
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = 1
    batch_count: int = 1


class SupportedModel(Enum):
    "Enum of models supported by the API"

    AnythingV3 = "Linaqruf/anything-v3.0"
    StableDiffusion1_5 = "runwayml/stable-diffusion-v1-5"
    StableDiffusion1_4 = "CompVis/stable-diffusion-v1-4"


@dataclass
class Txt2ImgQueueEntry:
    "Dataclass for a queue entry"

    data: Txt2imgData
    model: SupportedModel
    scheduler: Scheduler
    backend: Literal["PyTorch", "TensorRT"]
