from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union
from uuid import uuid4

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_depth2img import (
    StableDiffusionDepth2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers


@dataclass
class Job:
    "Base class for all jobs"

    data: Any
    model: str
    websocket_id: Union[str, None] = field(default=None)
    save_image: bool = field(default=True)


@dataclass
class ImageMetadata:
    "Metadata written to the image when it is saved"

    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: str
    model: str


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
    batch_size: int = 1
    batch_count: int = 1


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
    batch_size: int = 1
    batch_count: int = 1
    strength: float = 0.6


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
    batch_size: int = 1
    batch_count: int = 1


@dataclass
class ImageVariationsData:
    "Dataclass for the data of an img2img request"

    image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = 1
    batch_count: int = 1


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
]
