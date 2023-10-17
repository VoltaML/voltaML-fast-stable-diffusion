from contextlib import ExitStack

from core.config import config


def is_hypertile_available():
    "Checks whether hypertile is available"
    try:
        import hyper_tile  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


if is_hypertile_available():
    from hyper_tile import split_attention  # pylint: disable=unused-import


def hypertile(unet, height: int, width: int) -> ExitStack:
    from hyper_tile import split_attention  # noqa: F811

    s = ExitStack()
    s.enter_context(
        split_attention(unet, height / width, tile_size=config.api.hypertile_unet_chunk)
    )
    return s
