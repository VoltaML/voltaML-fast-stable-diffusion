import logging
import multiprocessing
from dataclasses import Field, dataclass, field, fields
from typing import Literal, Union

from dataclasses_json import CatchAll, DataClassJsonMixin, Undefined, dataclass_json
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import ControlNetMode

logger = logging.getLogger(__name__)


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 25
    batch_count: int = 1
    batch_size: int = 1


@dataclass
class Img2ImgConfig:
    "Configuration for the image to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 25
    batch_count: int = 1
    batch_size: int = 1
    resize_method: int = 0
    denoising_strength: float = 0.6


@dataclass
class InpaintingConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg_scale: int = 7
    seed: int = -1
    batch_count: int = 1
    batch_size: int = 1
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value


@dataclass
class ControlNetConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: int = 7
    steps: int = 25
    batch_count: int = 1
    batch_size: int = 1
    sampler: int = KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler.value
    controlnet: ControlNetMode = ControlNetMode.CANNY
    controlnet_conditioning_scale: float = 1.0
    detection_resolution: int = 512


@dataclass
class RealESRGANConfig:
    "Configuration for the RealESRGAN upscaler"

    model: str = "RealESRGAN_x4plus_anime_6B"
    scale_factor: int = 4


@dataclass
class APIConfig:
    "Configuration for the API"

    websocket_sync_interval: float = 0.02
    websocket_perf_interval: float = 1.0
    attention_processor: Literal["xformers", "spda"] = "xformers"
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "disabled"
    use_tomesd: bool = False
    tomesd_ratio: float = 0.4
    tomesd_downsample_layers: Literal[1, 2, 4, 8] = 1
    channels_last: bool = True
    vae_slicing: bool = True
    trace_model: bool = False
    offload: Literal["module", "model", "disabled"] = "disabled"
    image_preview_delay: float = 2.0
    device_id: int = 0
    device_type: Literal["cpu", "cuda", "mps", "directml"] = "cuda"
    use_fp32: bool = False

    @property
    def device(self):
        "Return the device string"

        if self.device_type == "cpu":
            return "cpu"
        if self.device_type == "directml":
            raise NotImplementedError("DirectML is not supported yet")

        return f"{self.device_type}:{self.device_id}"


@dataclass
class AITemplateConfig:
    "Configuration for model inference and acceleration"

    num_threads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


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

    # set to "Salesforce/blip-image-captioning-base" for an extra vram
    caption_model: str = "Salesforce/blip-image-captioning-large"
    visualizer_model: str = "ViT-L-14/openai"

    offload_captioner: bool = (
        False  # should net a very big vram save for minimal performance cost
    )
    offload_visualizer: bool = False  # should net a somewhat big vram save for a bigger performance cost compared to captioner

    chunk_size: int = 2048  # set to 1024 for lower vram usage
    flavor_intermediate_count: int = 2048  # set to 1024 for lower vram usage

    flamingo_model: str = "dhansmair/flamingo-mini"

    caption_max_length: int = 32


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Configuration(DataClassJsonMixin):
    "Main configuration class for the application"

    txt2img: Txt2ImgConfig = field(default=Txt2ImgConfig())
    img2img: Img2ImgConfig = field(default=Img2ImgConfig())
    inpainting: InpaintingConfig = field(default=InpaintingConfig())
    controlnet: ControlNetConfig = field(default=ControlNetConfig())
    api: APIConfig = field(default=APIConfig())
    interrogator: InterrogatorConfig = field(default=InterrogatorConfig())
    aitemplate: AITemplateConfig = field(default=AITemplateConfig())
    bot: BotConfig = field(default=BotConfig())
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
