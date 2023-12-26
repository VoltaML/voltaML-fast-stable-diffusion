from dataclasses import dataclass, field
from typing import Dict, Literal, Union

from dataclasses_json.api import DataClassJsonMixin
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import SigmaScheduler

LatentScaleModel = Literal[
    "nearest",
    "area",
    "bilinear",
    "bislerp",
    "bicubic",
    "nearest-exact",
]


@dataclass
class Flag:
    "Base class for all flags"


@dataclass
class HighResFixFlag(Flag, DataClassJsonMixin):
    "Flag to fix high resolution images"

    enabled: bool = False  # For storing in json

    scale: float = 2
    mode: Literal["latent", "image"] = "latent"

    # Image Upscaling
    image_upscaler: str = "RealESRGAN_x4plus_anime_6B"

    # Latent Upscaling
    latent_scale_mode: LatentScaleModel = "bislerp"
    antialiased: bool = False

    # Img2img
    strength: float = 0.7
    steps: int = 50


@dataclass
class DeepshrinkFlag(Flag, DataClassJsonMixin):
    "Flag for deepshrink"

    enabled: bool = False  # For storing in json

    depth_1: int = 3  # -1 to 12; steps of 1
    stop_at_1: float = 0.15  # 0 to 0.5; steps of 0.01

    depth_2: int = 4  # -1 to 12; steps of 1
    stop_at_2: float = 0.30  # 0 to 0.5; steps of 0.01

    scaler: LatentScaleModel = "bislerp"
    base_scale: float = 0.5  # 0.05 to 1.0; steps of 0.05
    early_out: bool = False


@dataclass
class ScalecrafterFlag(Flag, DataClassJsonMixin):
    "Flag for Scalecrafter settings"

    enabled: bool = False  # For storing in json

    base: str = "sd15"
    # In other words: allow untested/"unsafe" resolutions like "1234x4321"
    unsafe_resolutions: bool = True
    # May produce more "appealing" images, but will triple, or even quadruple memory usage.
    disperse: bool = False


@dataclass
class XLOriginalSize:
    width: int = 1024
    height: int = 1024


@dataclass
class SDXLFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL settings"

    original_size: XLOriginalSize = field(default_factory=XLOriginalSize)


@dataclass
class SDXLRefinerFlag(Flag, DataClassJsonMixin):
    "Flag for SDXL refiners"

    steps: int = 50
    strength: float = 0.3
    model: str = ""
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5


@dataclass
class UpscaleFlag(Flag, DataClassJsonMixin):
    "Flag for upscaling"

    enabled: bool = False  # For storing in json

    upscale_factor: float = field(default=4)
    tile_size: int = field(default=128)
    tile_padding: int = field(default=10)
    model: str = field(default="RealESRGAN_x4plus_anime_6B")


@dataclass
class ADetailerFlag(Flag, DataClassJsonMixin):
    "Flag for ADetailer settings"

    enabled: bool = field(default=False)  # For storing in json

    # Inpainting
    image: Union[bytes, str, None] = field(default=None)
    mask_image: Union[bytes, str, None] = field(default=None)
    scheduler: Union[
        int, str
    ] = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    steps: int = field(default=40)
    cfg_scale: float = field(default=7)
    self_attention_scale: float = field(default=1.0)
    sigmas: SigmaScheduler = field(default="automatic")
    seed: int = field(default=0)
    strength: float = field(default=0.45)
    sampler_settings: Dict = field(default_factory=dict)
    prompt_to_prompt_settings: Dict = field(default_factory=dict)

    # ADetailer specific
    mask_dilation: int = field(default=4)
    mask_blur: int = field(default=4)
    mask_padding: int = field(default=32)
