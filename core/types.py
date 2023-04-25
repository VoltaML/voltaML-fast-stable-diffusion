from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

InferenceBackend = Literal["PyTorch", "TensorRT", "AITemplate", "ONNX"]
Backend = Literal["PyTorch", "TensorRT", "AITemplate", "unknown", "LoRA"]


@dataclass
class Job:
    "Base class for all jobs"

    data: Any
    model: str
    websocket_id: Union[str, None] = field(default=None)
    save_image: Literal[True, False, "r2"] = True
    save_grid: bool = False
    flags: Dict[str, Dict] = field(default_factory=dict)


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
    "Enum of modes for the ControlNet"

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
class InterrogationData:
    "Dataclass for the data of an interrogation request"

    image: Union[bytes, str]
    caption: Optional[str] = field(default=None)
    treshold: float = field(default=0.5)
    id: str = field(default_factory=lambda: uuid4().hex)


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
class RealESRGanData:
    "Dataclass for the data of a real esrgan request"

    image: Union[bytes, str]
    id: str = field(default_factory=lambda: uuid4().hex)
    upscale_factor: int = field(default=4)


@dataclass
class SDUpscaleData:
    "Dataclass for the data of Stable Diffusion Tiled Upscale request"

    prompt: str
    image: Union[bytes, str]
    scheduler: KarrasDiffusionSchedulers
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    tile_size: int = field(default=128)
    tile_border: int = field(default=32)
    original_image_slice: int = field(default=32)
    noise_level: int = field(default=40)


@dataclass
class InterrogatorQueueEntry(Job):
    "Dataclass for an interrogation queue entry"

    data: InterrogationData


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
class ControlNetQueueEntry(Job):
    "Dataclass for a control net queue entry"

    data: ControlNetData


@dataclass
class RealESRGANQueueEntry(Job):
    "Dataclass for a real esrgan job"

    data: RealESRGanData


@dataclass
class SDUpscaleQueueEntry(Job):
    "Dataclass for a stable diffusion upscale job"

    data: SDUpscaleData
    model: str = field(default="stabilityai/stable-diffusion-x4-upscaler")


@dataclass
class TRTBuildRequest:
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


@dataclass
class QuantizationDict:
    "Dataclass for quantization parameters"

    vae_encoder: Literal[True, False, None] = None
    vae_decoder: Literal[True, False, None] = None
    unet: Literal[True, False, None] = None
    text_encoder: Literal[True, False, None] = None


@dataclass
class ONNXBuildRequest:
    "Dataclass for requesting a build of an ONNX engine"

    model_id: str
    simplify_unet: bool = False
    quant_dict: QuantizationDict = field(default_factory=QuantizationDict)


@dataclass
class ConvertModelRequest:
    "Dataclass for requesting a conversion of a model"

    model: str
    use_fp32: bool = False
    safetensors: bool = False


PyTorchModelType = Union[
    DiffusionPipeline,
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


@dataclass
class ModelResponse:
    "Dataclass for a response containing a loaded model info"

    name: str
    path: str
    backend: Backend
    valid: bool
    state: Literal["loading", "loaded", "not loaded"] = field(default="not loaded")
    loras: List[str] = field(default_factory=list)
