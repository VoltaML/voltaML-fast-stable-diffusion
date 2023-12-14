from dataclasses import dataclass, field
from typing import Literal, List

from dataclasses_json.api import DataClassJsonMixin

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
    antialiased: bool = False


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
class AnimateDiffFlag(Flag, DataClassJsonMixin):
    "Flag for AnimateDiff"

    motion_model: str = ""
    frames: int = 16
    fps: int = 10  # not working

    # https://github.com/TianxingWu/FreeInit/blob/master/examples/AnimateDiff/animatediff/pipelines/pipeline_animation.py#L470
    # Probably self-explanatory, but increases generation time to ${freeinit_iterations}x.
    freeinit_iterations: int = -1  # -1 to disable, 5 recommended
    freeinit_fast_sampling: bool = False  # ruins quality, speeds up inbetween iterations.

    # Big maybes:
    # - https://github.com/omerbt/TokenFlow
    # - https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/commit/5bbcae4d226e8f298a8b204e9cc9b2dd41fbe417

    # TODO: steal code from here:
    # - https://github.com/guoyww/AnimateDiff/pull/132/files
    # - https://github.com/guoyww/AnimateDiff/pull/25/files
    # - https://github.com/guoyww/AnimateDiff/pull/8/files


    # only active when frames > 16 --> sliding context window.
    context_size: int = 16
    frame_stride: int = 1
    frame_overlap: int = 4
    context_scheduler: Literal[
        "uniform", "uniform_constant", "uniform_v2"
    ] = "uniform_v2"

    closed_loop: bool = True

    # increase processing time for decreased memory usage
    chunk_feed_forward: bool = True  # probably not working

    input_video: str = ""  # not working
    init_image: str = ""  # not working
    video_controlnets: List[str] = field(default_factory=list)  # not working
