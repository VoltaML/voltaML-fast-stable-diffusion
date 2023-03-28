import os
from pathlib import Path

from .config import (
    Configuration,
    Img2ImgConfig,
    Txt2ImgConfig,
    load_config,
    save_config,
)

config = load_config()
config.api.cache_dir = os.environ.get("DIFFUSERS_CACHE", config.api.cache_dir)

# Create cache directory if it doesn't exist
Path(config.api.cache_dir).mkdir(parents=True, exist_ok=True)

__all__ = [
    "config",
    "Img2ImgConfig",
    "Txt2ImgConfig",
    "Configuration",
    "save_config",
    "load_config",
]
