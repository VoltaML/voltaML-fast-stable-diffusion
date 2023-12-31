from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE as DIFFUSERS_CACHE

from ._config import (
    Configuration,
    load_config,
    save_config,
)
from .default_settings import (
    Txt2ImgConfig,
    Img2ImgConfig,
)

config = load_config()

# Create cache directory if it doesn't exist
Path(DIFFUSERS_CACHE).mkdir(parents=True, exist_ok=True)

__all__ = [
    "config",
    "Img2ImgConfig",
    "Txt2ImgConfig",
    "Configuration",
    "save_config",
    "load_config",
]
