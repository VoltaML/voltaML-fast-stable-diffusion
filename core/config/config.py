import logging
import multiprocessing
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.constants import DIFFUSERS_CACHE

from core.types import ControlNetMode

logger = logging.getLogger(__name__)


@dataclass
class Txt2ImgConfig:
    "Configuration for the text to image pipeline"

    width: int = 512
    height: int = 512
    seed: int = -1
    cfgScale: int = 7
    sampler: int = KarrasDiffusionSchedulers.UniPCMultistepScheduler.value
    prompt: str = ""
    negativePrompt: str = ""
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
    sampler: int = KarrasDiffusionSchedulers.UniPCMultistepScheduler.value
    prompt: str = ""
    negativePrompt: str = ""
    steps: int = 25
    batchCount: int = 1
    batchSize: int = 1
    resizeMethod: int = 0
    denoisingStrength: float = 0.6


@dataclass
class ImageVariationsConfig:
    "Configuration for the image variations pipeline"

    batchCount: int = 1
    batchSize: int = 1
    cfgScale: int = 7
    seed: int = -1
    sampler: int = KarrasDiffusionSchedulers.UniPCMultistepScheduler.value
    steps: int = 25


@dataclass
class InpaintingConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negativePrompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    cfgScale: int = 7
    seed: int = -1
    batchCount: int = 1
    batchSize: int = 1
    sampler: int = KarrasDiffusionSchedulers.UniPCMultistepScheduler.value


@dataclass
class ControlNetConfig:
    "Configuration for the inpainting pipeline"

    prompt: str = ""
    negativePrompt: str = ""
    width: int = 512
    height: int = 512
    seed: int = -1
    cfgScale: int = 7
    steps: int = 25
    batchCount: int = 1
    batchSize: int = 1
    sampler: int = KarrasDiffusionSchedulers.UniPCMultistepScheduler.value
    controlnet: ControlNetMode = ControlNetMode.CANNY
    controlnetConditioningScale: float = 1.0
    detectionResolution: int = 512


@dataclass
class APIConfig:
    "Configuration for the API"

    websocketSyncInterval = 0.02
    websocketPerfInterval = 1
    cache_dir: str = field(default=DIFFUSERS_CACHE)
    lowVRAM: bool = False


@dataclass
class AITemplateConfig:
    "Configuration for model inference and acceleration"

    numThreads: int = field(default=min(multiprocessing.cpu_count() - 1, 8))


@dataclass
class BotConfig:
    "Configuration for the bot"

    defaultScheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.UniPCMultistepScheduler
    )
    verbose: bool = False
    useDefaultNegativePrompt: bool = True


@dataclass
class Configuration(DataClassJsonMixin):
    "Main configuration class for the application"

    txt2img: Txt2ImgConfig = field(default=Txt2ImgConfig())
    img2img: Img2ImgConfig = field(default=Img2ImgConfig())
    imageVariations: ImageVariationsConfig = field(default=ImageVariationsConfig())
    inpainting: InpaintingConfig = field(default=InpaintingConfig())
    controlnet: ControlNetConfig = field(default=ControlNetConfig())
    api: APIConfig = field(default=APIConfig())
    aitemplate: AITemplateConfig = field(default=AITemplateConfig())
    bot: BotConfig = field(default=BotConfig())


def save_config(config: Configuration):
    "Save the configuration to a file"

    logger.info("Saving configuration to config.json")

    with open("config.json", "w", encoding="utf-8") as f:
        f.write(config.to_json(ensure_ascii=False, indent=4))


def load_config():
    "Load the configuration from a file"

    logger.info("Loading configuration from config.json")

    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = Configuration.from_json(f.read())
            logger.info("Configuration loaded from config.json")
            return config

    except FileNotFoundError:
        logger.info("config.json not found, creating a new one")
        config = Configuration()
        save_config(config)
        logger.info("Configuration saved to config.json")
        return config
