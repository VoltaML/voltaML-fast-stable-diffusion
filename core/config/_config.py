import logging
from dataclasses import Field, dataclass, field, fields

from dataclasses_json import CatchAll, DataClassJsonMixin, Undefined, dataclass_json

from core.config.samplers.sampler_config import SamplerConfig
from .api_settings import APIConfig
from .bot_settings import BotConfig
from .default_settings import (
    Txt2ImgConfig,
    Img2ImgConfig,
    InpaintingConfig,
    ControlNetConfig,
    UpscaleConfig,
    AITemplateConfig,
    ONNXConfig,
)
from .frontend_settings import FrontendConfig
from .interrogator_settings import InterrogatorConfig

logger = logging.getLogger(__name__)


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
