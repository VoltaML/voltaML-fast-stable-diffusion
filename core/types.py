from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

InferenceBackend = Literal["PyTorch", "AITemplate", "SDXL", "ONNX"]
SigmaScheduler = Literal["automatic", "karras", "exponential", "polyexponential", "vp"]
Backend = Literal[
    "PyTorch",
    "SDXL",
    "AITemplate",
    "unknown",
    "LoRA",
    "LyCORIS",
    "Textual Inversion",
    "ONNX",
    "VAE",
    "Upscaler",
    "GPT",  # for prompt-expansion
]
PyTorchModelBase = Literal[
    "SD1.x", "SD2.x", "SDXL", "Kandinsky 2.1", "Kandinsky 2.2", "Wuerstchen", "IF"
]
PyTorchModelStage = Literal["text_encoding", "first_stage", "last_stage"]
ImageFormats = Literal["png", "jpeg", "webp"]


@dataclass
class Job:
    "Base class for all jobs"

    data: Any
    model: str
    websocket_id: Union[str, None] = field(default=None)
    save_image: Literal[True, False, "r2"] = True
    flags: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class InterrogationData:
    "Dataclass for the data of an interrogation request"

    image: Union[bytes, str]
    caption: Optional[str] = field(default=None)
    threshold: float = field(default=0.5)
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Txt2imgData:
    "Dataclass for the data of a txt2img request"

    prompt: str
    scheduler: Union[str, KarrasDiffusionSchedulers]
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    self_attention_scale: float = field(default=0.0)
    sigmas: SigmaScheduler = field(default="automatic")
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    sampler_settings: Dict = field(default_factory=dict)
    prompt_to_prompt_settings: Dict = field(default_factory=dict)


@dataclass
class Img2imgData:
    "Dataclass for the data of an img2img request"

    prompt: str
    image: Union[bytes, str]
    scheduler: Union[str, KarrasDiffusionSchedulers]
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    self_attention_scale: float = field(default=0.0)
    sigmas: SigmaScheduler = field(default="automatic")
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    strength: float = field(default=0.6)
    sampler_settings: Dict = field(default_factory=dict)
    prompt_to_prompt_settings: Dict = field(default_factory=dict)


@dataclass
class InpaintData:
    "Dataclass for the data of an img2img request"

    prompt: str
    image: Union[bytes, str]
    mask_image: Union[bytes, str]
    scheduler: Union[str, KarrasDiffusionSchedulers]
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    self_attention_scale: float = field(default=0.0)
    sigmas: SigmaScheduler = field(default="automatic")
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    sampler_settings: Dict = field(default_factory=dict)
    prompt_to_prompt_settings: Dict = field(default_factory=dict)


@dataclass
class ControlNetData:
    "Dataclass for the data of a control net request"

    prompt: str
    image: Union[bytes, str]
    scheduler: Union[str, KarrasDiffusionSchedulers]
    controlnet: str
    id: str = field(default_factory=lambda: uuid4().hex)
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    sigmas: SigmaScheduler = field(default="automatic")
    seed: int = field(default=0)
    batch_size: int = field(default=1)
    batch_count: int = field(default=1)
    controlnet_conditioning_scale: float = field(default=1.0)
    detection_resolution: int = field(default=512)
    sampler_settings: Dict = field(default_factory=dict)
    prompt_to_prompt_settings: Dict = field(default_factory=dict)

    canny_low_threshold: int = field(default=100)
    canny_high_threshold: int = field(default=200)

    mlsd_thr_v: float = field(default=0.1)
    mlsd_thr_d: float = field(default=0.1)

    is_preprocessed: bool = field(default=False)
    save_preprocessed: bool = field(default=False)
    return_preprocessed: bool = field(default=True)


@dataclass
class UpscaleData:
    "Dataclass for the data of a real esrgan request"

    image: Union[bytes, str]
    id: str = field(default_factory=lambda: uuid4().hex)
    upscale_factor: float = field(default=4)
    tile_size: int = field(default=128)
    tile_padding: int = field(default=10)


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
class UpscaleQueueEntry(Job):
    "Dataclass for a real esrgan job"

    data: UpscaleData


@dataclass
class QuantizationDict:
    "Dataclass for quantization parameters"

    vae_encoder: Literal["no-quant", "uint8", "int8"] = "no-quant"
    vae_decoder: Literal["no-quant", "uint8", "int8"] = "no-quant"
    unet: Literal["no-quant", "uint8", "int8"] = "no-quant"
    text_encoder: Literal["no-quant", "uint8", "int8"] = "no-quant"


@dataclass
class ONNXBuildRequest:
    "Dataclass for requesting a build of an ONNX engine"

    model_id: str
    simplify_unet: bool = False
    convert_to_fp16: bool = False
    quant_dict: QuantizationDict = field(default_factory=QuantizationDict)


@dataclass
class ConvertModelRequest:
    "Dataclass for requesting a conversion of a model"

    model: str
    safetensors: bool = False


PyTorchModelType = Union[
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
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
class AITemplateDynamicBuildRequest:
    "Dataclass for requesting a build of an engine"

    model_id: str
    width: Tuple[int, int] = field(default=(64, 2048))
    height: Tuple[int, int] = field(default=(64, 2048))
    batch_size: Tuple[int, int] = field(default=(1, 4))
    clip_chunks: int = field(default=6)
    threads: Optional[int] = field(default=None)


@dataclass
class ModelResponse:
    "Dataclass for a response containing a loaded model info"

    name: str
    path: str
    backend: Backend
    valid: bool
    vae: str
    state: Literal["loading", "loaded", "not loaded"] = field(default="not loaded")
    textual_inversions: List[str] = field(default_factory=list)
    type: PyTorchModelBase = "SD1.x"
    stage: PyTorchModelStage = "last_stage"


@dataclass
class LoraLoadRequest:
    "Dataclass for loading a LoRA onto a model"

    model: str
    lora: str
    weight: float = 0.5


@dataclass
class LoraUnloadRequest:
    "Dataclass for unloading a LoRA from a model"

    model: str
    lora: str


@dataclass
class VaeLoadRequest:
    "Dataclass for loading a VAE into a model"

    model: str
    vae: str


@dataclass
class TextualInversionLoadRequest:
    "Dataclass for loading a textual inversion onto a model"

    model: str
    textual_inversion: str


@dataclass
class DeleteModelRequest:
    "Dataclass for requesting a deletion of a model"

    model_path: str
    model_type: Literal[
        "models", "lora", "textual-inversion", "lycoris", "vae", "aitemplate"
    ]


@dataclass
class Capabilities:
    "Dataclass for capabilities of a GPU"

    # ["cpu", "cuda", "directml", "mps", "xpu", "vulkan"]
    supported_backends: List[List[str]] = field(
        default_factory=lambda: [["CPU", "cpu"]]
    )
    # ["float16", "float32", "bfloat16"]
    supported_precisions_gpu: List[str] = field(default_factory=lambda: ["float32"])
    # ["float16", "float32", "bfloat16"]
    supported_precisions_cpu: List[str] = field(default_factory=lambda: ["float32"])

    supported_torch_compile_backends: List[str] = field(
        default_factory=lambda: ["inductor"]
    )

    supported_self_attentions: List[List[str]] = field(
        default_factory=lambda: [
            ["Cross-Attention", "cross-attention"],
            ["Subquadratic Attention", "subquadratic"],
            ["Multihead Attention", "multihead"],
        ]
    )

    # Does he have bitsandbytes installed?
    supports_int8: bool = False

    # Does the current build support xformers?
    # Useful for e.g. torch nightlies
    supports_xformers: bool = False

    # Needed for sfast.
    supports_triton: bool = False

    # Volta+ (>=7.0)
    has_tensor_cores: bool = True

    # Ampere+ (>=8.6)
    has_tensorfloat: bool = False

    hypertile_available: bool = False


InferenceJob = Union[
    Txt2ImgQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    ControlNetQueueEntry,
]
