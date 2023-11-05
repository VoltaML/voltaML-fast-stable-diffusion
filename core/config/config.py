import logging
import multiprocessing
from dataclasses import Field, dataclass, field, fields
from typing import Dict, List, Literal, Optional, Union

import torch
from dataclasses_json import CatchAll, DataClassJsonMixin, Undefined, dataclass_json
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.config.samplers.sampler_config import SamplerConfig
from core.flags import HighResFixFlag
from core.types import SigmaScheduler

logger = logging.getLogger(__name__)


@dataclass
class BaseDiffusionMixin:
    width: int = 512
    height: int = 512
    batch_count: int = 1
    batch_size: int = 1
    seed: int = -1
    cfg_scale: int = 7
    steps: int = 40
    prompt: str = ""
    negative_prompt: str = ""
    sampler: Union[
        int, str
    ] = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    sigmas: SigmaScheduler = "automatic"


@dataclass
class QuantDict:
    vae_decoder: Optional[bool] = None
    vae_encoder: Optional[bool] = None
    unet: Optional[bool] = None
    text_encoder: Optional[bool] = None


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


@dataclass
class ControlNetConfig(BaseDiffusionMixin):
    "Configuration for the inpainting pipeline"

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
class APIConfig:
    "Configuration for the API"

    # Autoload
    autoloaded_textual_inversions: List[str] = field(default_factory=list)
    autoloaded_models: List[str] = field(default_factory=list)
    autoloaded_vae: Dict[str, str] = field(default_factory=dict)

    # Websockets and intervals
    websocket_sync_interval: float = 0.02
    websocket_perf_interval: float = 1.0
    enable_websocket_logging: bool = True

    # TomeSD
    use_tomesd: bool = False  # really extreme, probably will have to wait around until tome improves a bit
    tomesd_ratio: float = 0.25  # had to tone this down, 0.4 is too big of a context loss even on short prompts
    tomesd_downsample_layers: Literal[1, 2, 4, 8] = 1

    # General optimizations
    autocast: bool = False
    attention_processor: Literal[
        "xformers", "sdpa", "cross-attention", "subquadratic", "multihead"
    ] = "sdpa"
    subquadratic_size: int = 512
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "disabled"
    channels_last: bool = True
    trace_model: bool = False
    clear_memory_policy: Literal["always", "after_disconnect", "never"] = "always"
    offload: Literal["module", "model", "disabled"] = "disabled"
    data_type: Literal["float32", "float16", "bfloat16"] = "float16"
    dont_merge_latents: bool = (
        False  # Will drop performance, but could help with some VRAM issues
    )

    # CUDA specific optimizations
    reduced_precision: bool = False
    cudnn_benchmark: bool = False
    deterministic_generation: bool = False

    # Device settings
    device: str = "cuda:0"

    # Critical
    enable_shutdown: bool = True

    # CLIP
    clip_skip: int = 1
    clip_quantization: Literal["full", "int8", "int4"] = "full"

    huggingface_style_parsing: bool = False

    # Saving
    save_path_template: str = "{folder}/{prompt}/{id}-{index}.{extension}"
    image_extension: Literal["png", "webp", "jpeg"] = "png"
    image_quality: int = 95
    image_return_format: Literal["bytes", "base64"] = "base64"

    # Grid
    disable_grid: bool = False

    # Torch compile
    torch_compile: bool = False
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
    ] = "reduce-overhead"

    sfast_compile: bool = False
    sfast_xformers: bool = True
    sfast_triton: bool = True
    sfast_cuda_graph: bool = True

    # Hypertile
    hypertile: bool = False
    hypertile_unet_chunk: int = 256

    # K_Diffusion
    sgm_noise_multiplier: bool = False  # also known as "alternate DDIM ODE"
    kdiffusers_quantization: bool = True  # improves sampling quality

    # "philox" is what a "cuda" generator would be, except, it's on cpu
    generator: Literal["device", "cpu", "philox"] = "device"

    # VAE
    live_preview_method: Literal["disabled", "approximation", "taesd"] = "approximation"
    live_preview_delay: float = 2.0
    vae_slicing: bool = True
    vae_tiling: bool = False

    # Prompt expansion (very, and I mean VERYYYY heavily inspired/copied from lllyasviel/Fooocus)
    prompt_to_prompt: bool = False
    prompt_to_prompt_model: Literal[
        "lllyasviel/Fooocus-Expansion",
        "daspartho/prompt-extend",
        "succinctly/text2image-prompt-generator",
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator",
    ] = "lllyasviel/Fooocus-Expansion"
    prompt_to_prompt_device: Literal["cpu", "gpu"] = "gpu"

    @property
    def dtype(self):
        "Return selected data type"
        if self.data_type == "bfloat16":
            return torch.bfloat16
        if self.data_type == "float16":
            return torch.float16
        return torch.float32

    @property
    def overwrite_generator(self) -> bool:
        "Whether the generator needs to be overwritten with 'cpu.'"

        return any(
            map(lambda x: x in self.device, ["mps", "directml", "vulkan", "intel"])
        )


@dataclass
class AITemplateConfig:
    "Configuration for model inference and acceleration"

    num_threads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


@dataclass
class ONNXConfig:
    "Configuration for ONNX acceleration"

    quant_dict: QuantDict = field(default_factory=QuantDict)


@dataclass
class BotConfig:
    "Configuration for the bot"

    default_scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler
    )
    verbose: bool = False
    use_default_negative_prompt: bool = True


@dataclass
class InterrogatorConfig:
    "Configuration for interrogation models"

    # set to "Salesforce/blip-image-captioning-base" for an extra gig of vram
    caption_model: str = "Salesforce/blip-image-captioning-large"
    visualizer_model: str = "ViT-L-14/openai"

    offload_captioner: bool = False
    offload_visualizer: bool = False

    chunk_size: int = 2048  # set to 1024 for lower vram usage
    flavor_intermediate_count: int = 2048  # set to 1024 for lower vram usage

    flamingo_model: str = "dhansmair/flamingo-mini"

    caption_max_length: int = 32


@dataclass
class FrontendConfig:
    "Configuration for the frontend"

    theme: str = "dark"
    background_image_override: str = ""
    enable_theme_editor: bool = False
    image_browser_columns: int = 5
    on_change_timer: int = 0
    nsfw_ok_threshold: int = 0
    disable_analytics: bool = False


@dataclass
class FlagsConfig:
    "Configuration for flags"

    highres: HighResFixFlag = field(default_factory=HighResFixFlag)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Configuration(DataClassJsonMixin):
    "Main configuration class for the application"

    txt2img: Txt2ImgConfig = field(default_factory=Txt2ImgConfig)
    img2img: Img2ImgConfig = field(default_factory=Img2ImgConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    api: APIConfig = field(default_factory=APIConfig)
    interrogator: InterrogatorConfig = field(default_factory=InterrogatorConfig)
    aitemplate: AITemplateConfig = field(default_factory=AITemplateConfig)
    onnx: ONNXConfig = field(default_factory=ONNXConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    flags: FlagsConfig = field(default_factory=FlagsConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    extra: CatchAll = field(default_factory=dict)


def save_config(config: Configuration):
    "Save the configuration to a file"

    logger.info("Saving configuration to data/settings.json")

    with open("data/settings.json", "w", encoding="utf-8") as f:
        f.write(config.to_json(ensure_ascii=False, indent=4))


def update_config(config: Configuration, new_config: Configuration):
    "Update the configuration with new values instead of overwriting the pointer"

    for cls_field in fields(new_config):
        assert isinstance(cls_field, Field)
        setattr(config, cls_field.name, getattr(new_config, cls_field.name))


def load_config():
    "Load the configuration from a file"

    logger.info("Loading configuration from data/settings.json")

    try:
        with open("data/settings.json", "r", encoding="utf-8") as f:
            config = Configuration.from_json(f.read())
            logger.info("Configuration loaded from data/settings.json")
            return config

    except FileNotFoundError:
        logger.info("data/settings.json not found, creating a new one")
        config = Configuration()
        save_config(config)
        logger.info("Configuration saved to data/settings.json")
        return config
