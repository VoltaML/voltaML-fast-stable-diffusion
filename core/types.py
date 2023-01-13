from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from uuid import uuid4


class Scheduler(Enum):
    "Enum of schedulers supported by the API"

    euler_a = "Euler A"
    ddim = "DDIM"
    heun = "Heun"
    dpm_discrete = "DPM Discrete"
    dpm_ancestral = "DPM A"
    lms = "LMS"
    pndm = "PNDM"
    euler = "Euler"
    dpmpp_sde_ancestral = "DPMPP SDE A"
    dpmpp_2m = "DPMPP 2M"
    default = "Default"


@dataclass
class Txt2imgData:
    "Dataclass for the data of a txt2img request"

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
    StableDiffusion2_1 = "stabilityai/stable-diffusion-2-1"
    OpenJourney = "prompthero/openjourney"
    DreamlikeDiffusion = "dreamlike-art/dreamlike-diffusion-1.0"
    DreamlikePhotoreal = "dreamlike-art/dreamlike-photoreal-2.0"
    Protogen5_8_Anime = "darkstorm2150/Protogen_x5.8_Official_Release"
    SynthWave = "ItsJayQz/SynthwavePunk-v2"
    InkpunkDiffusion = "Envvi/Inkpunk-Diffusion"
    Protogen5_3_Realism = "darkstorm2150/Protogen_v5.3_Official_Release"


@dataclass
class Txt2ImgQueueEntry:
    "Dataclass for a queue entry"

    data: Txt2imgData
    model: SupportedModel
    scheduler: Scheduler
    backend: Literal["PyTorch", "TensorRT"]
    autoload: bool = field(default=False)
