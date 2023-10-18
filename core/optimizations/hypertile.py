from contextlib import ExitStack
from importlib.util import find_spec

from core.config import config


def is_hypertile_available():
    "Checks whether hypertile is available"
    return find_spec("hyper_tile") is not None


def hypertile(unet, height: int, width: int) -> ExitStack:
    from hyper_tile import split_attention  # noqa: F811

    s = ExitStack()
    s.enter_context(
        split_attention(unet, height / width, tile_size=config.api.hypertile_unet_chunk)
    )
    return s
