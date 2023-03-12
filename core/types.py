from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

InferenceBackend = Literal["PyTorch", "TensorRT", "AITemplate"]


@dataclass
class Job:
    "Base class for all jobs"

    data: Any
    model: str
    websocket_id: Union[str, None] = field(default=None)
    save_image: bool = field(default=True)


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
    AnythingV4 = "andite/anything-v4.0"


class ControlNetMode(Enum):
    "Enum of modes for the control net"

    CANNY = "lllyasviel/sd-controlnet-canny"
    DEPTH = "lllyasviel/sd-controlnet-depth"
    HED = "lllyasviel/sd-controlnet-hed"
    MLSD = "lllyasviel/sd-controlnet-mlsd"
    NORMAL = "lllyasviel/sd-controlnet-normal"
    OPENPOSE = "lllyasviel/sd-controlnet-openpose"
    SCRIBBLE = "lllyasviel/sd-controlnet-scribble"
    SEGMENTATION = "lllyasviel/sd-controlnet-seg"
    NONE = "none"


@dataclass
class Txt2imgData:
    "Dataclass for the data of a txt2img request"

    prompt: str
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)


@dataclass
class Img2imgData:
    "Dataclass for the data of an img2img request"

    prompt: str
    image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    strength: float = field(default=0.6)


@dataclass
class InpaintData:
    "Dataclass for the data of an img2img request"

    prompt: str
    image: Union[bytes, str]
    mask_image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)


@dataclass
class ImageVariationsData:
    "Dataclass for the data of an img2img request"

    image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)


@dataclass
class ControlNetData:
    "Dataclass for the data of a control net request"

    prompt: str
    image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    controlnet: ControlNetMode
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    controlnet_conditioning_scale: float = field(default=1.0)
    detection_resolution: int = field(default=512)

    canny_low_threshold: int = field(default=100)
    canny_high_threshold: int = field(default=200)

    mlsd_thr_v: float = field(default=0.1)
    mlsd_thr_d: float = field(default=0.1)


@dataclass
class Txt2ImgQueueEntry(Job):
    "Dataclass for a text to image queue entry"

    data: Txt2imgData


@dataclass
class Img2ImgQueueEntry(Job):
    "Dataclass for an image to image queue entry"

    data: Img2imgData


@dataclass
class InpaintQueueEntry(Job):
    "Dataclass for an image to image queue entry"

    data: InpaintData


@dataclass
class ImageVariationsQueueEntry(Job):
    "Dataclass for an image to image queue entry"

    data: ImageVariationsData


@dataclass
class ControlNetQueueEntry(Job):
    "Dataclass for a control net queue entry"

    data: ControlNetData


@dataclass
class BuildRequest:
    "Dataclass for requesting a build of an engine"

    model_id: str
    subfolder: str = ""
    hf_token: str = ""
    fp16: bool = True
    verbose: bool = True
    opt_image_height: int = 512
    opt_image_width: int = 512
    max_batch_size: int = 1
    onnx_opset: int = 16
    build_static_batch: bool = False
    build_dynamic_shape: bool = True
    build_preview_features: bool = False
    force_engine_build: bool = False
    force_onnx_export: bool = False
    force_onnx_optimize: bool = False
    onnx_minimal_optimization: bool = False


PyTorchModelType = Union[
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionControlNetPipeline,
]


@dataclass
class AITemplateBuildRequest:
    "Dataclass for requesting a build of an engine"

    model_id: str
    width: int = field(default=512)
    height: int = field(default=512)
    batch_size: int = field(default=1)
    threads: Optional[int] = field(default=None)
