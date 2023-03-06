import multiprocessing
from dataclasses import dataclass, field
from typing import Literal

from dataclasses_json import DataClassJsonMixin
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.constants import DIFFUSERS_CACHE


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfgScale: int = 7
    sampler: int = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 25
    batchCount: int = 1
    batchSize: int = 1


@dataclass
class Img2ImgConfig:
    "Configuration for the image to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfgScale: int = 7
    sampler: int = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 25
    batchCount: int = 1
    batchSize: int = 1
    resizeMethod: int = 0
    denoisingStrength: float = 0.6


@dataclass
class ImageVariations:
    "Configuration for the image variations pipeline"

    batchCount: int = 1
    batchSize: int = 1
    cfgScale: int = 7
    seed: int = -1
    sampler: int = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler.value
    steps: int = 25


@dataclass
class Inpainting:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    cfgScale: int = 7
    seed: int = -1
    batchCount: int = 1
    batchSize: int = 1
    sampler: int = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler.value


@dataclass
class APIConfig:
    "Configuration for the API"

    websocket_sync_interval = 0.02
    websocket_perf_interval = 1


@dataclass
class InferenceConfig:
    "Configuration for model inference and acceleration"

    num_threads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


@dataclass
class Configuration(DataClassJsonMixin):
    "Main configuration class for the application"

    backend: Literal["PyTorch", "TensorRT", "AITemplate"] = "PyTorch"
    model: str = "none:PyTorch"
    txt2img: Txt2ImgConfig = field(default=Txt2ImgConfig())
    img2img: Img2ImgConfig = field(default=Img2ImgConfig())
    api: APIConfig = field(default=APIConfig())
    inference: InferenceConfig = field(default=InferenceConfig())
    imageVariations: ImageVariations = field(default=ImageVariations())
    inpainting: Inpainting = field(default=Inpainting())
    cache_dir: str = field(default=DIFFUSERS_CACHE)
