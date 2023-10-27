import logging
from dataclasses import dataclass, field

from dataclasses_json.api import DataClassJsonMixin
from diffusers.schedulers import KarrasDiffusionSchedulers

logger = logging.getLogger(__name__)


@dataclass
class Config(DataClassJsonMixin):
    "Dataclass that will store the configuration for the bot"

    supported_models: dict[str, str] = field(default_factory=dict)
    prefix: str = "!"

    extra_prompt: str = ""
    extra_negative_prompt: str = ""

    max_width: int = 1920
    max_height: int = 1920
    max_count: int = 4
    max_steps: int = 50

    default_width: int = 512
    default_height: int = 512
    default_count: int = 1
    default_steps: int = 30
    default_scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.DPMSolverMultistepScheduler
    )
    default_cfg: float = 7.0
    default_verbose: bool = False


def save_config(config: Config):
    "Save the configuration to a file"

    logger.info("Saving configuration to data/bot.json")

    with open("data/bot.json", "w", encoding="utf-8") as f:
        f.write(config.to_json(ensure_ascii=False, indent=4))


def load_config():
    "Load the configuration from a file"

    logger.info("Loading configuration from data/bot.json")

    try:
        with open("data/bot.json", "r", encoding="utf-8") as f:
            config = Config.from_json(f.read())
            logger.info("Configuration loaded from data/bot.json")
            return config

    except FileNotFoundError:
        logger.info("data/bot.json not found, creating a new one")
        config = Config()
        save_config(config)
        logger.info("Configuration saved to data/bot.json")
        return config
