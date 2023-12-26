import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Union

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.flags import (
    ADetailerFlag,
    DeepshrinkFlag,
    HighResFixFlag,
    ScalecrafterFlag,
    UpscaleFlag,
)
from core.types import SigmaScheduler


@dataclass
class QuantDict:
    "Configuration for ONNX quantization"

    vae_decoder: Optional[bool] = None
    vae_encoder: Optional[bool] = None
    unet: Optional[bool] = None
    text_encoder: Optional[bool] = None


@dataclass
class BaseDiffusionMixin:
    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    steps: int = 40
    prompt: str = ""
    negative_prompt: str = ""
    sampler: Union[
        int, str
    ] = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    sigmas: SigmaScheduler = "automatic"
    batch_count: int = 1
    batch_size: int = 1

    # Flags
    highres: HighResFixFlag = field(default_factory=HighResFixFlag)
    upscale: UpscaleFlag = field(default_factory=UpscaleFlag)
    deepshrink: DeepshrinkFlag = field(default_factory=DeepshrinkFlag)
    scalecrafter: ScalecrafterFlag = field(default_factory=ScalecrafterFlag)
    adetailer: ADetailerFlag = field(default_factory=ADetailerFlag)


@dataclass
class Txt2ImgConfig(BaseDiffusionMixin):
    "Configuration for the text to image pipeline"

    self_attention_scale: float = 0.0


@dataclass
class Img2ImgConfig(BaseDiffusionMixin):
    "Configuration for the image to image pipeline"

    resize_method: int = 0
    denoising_strength: float = 0.6
    self_attention_scale: float = 0.0


@dataclass
class InpaintingConfig(BaseDiffusionMixin):
    "Configuration for the inpainting pipeline"

    self_attention_scale: float = 0.0
    strength: float = 0.6


@dataclass
class ControlNetConfig(BaseDiffusionMixin):
    "Configuration for the inpainting pipeline"

    self_attention_scale: float = 0.0

    controlnet: str = "lllyasviel/sd-controlnet-canny"
    controlnet_conditioning_scale: float = 1.0
    detection_resolution: int = 512
    is_preprocessed: bool = False
    save_preprocessed: bool = False
    return_preprocessed: bool = True


@dataclass
class UpscaleConfig:
    "Configuration for the RealESRGAN upscaler"

    model: str = "RealESRGAN_x4plus_anime_6B"
    upscale_factor: int = 4
    tile_size: int = field(default=128)
    tile_padding: int = field(default=10)


@dataclass
class AITemplateConfig:
    "Configuration for model inference and acceleration"

    num_threads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


@dataclass
class ONNXConfig:
    "Configuration for ONNX acceleration"

    quant_dict: QuantDict = field(default_factory=QuantDict)
