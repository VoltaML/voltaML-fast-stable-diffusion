from contextlib import ExitStack

from core.config import config


def is_hypertile_available():
    "Checks whether hypertile is available"
    try:
        import hyper_tile
        return True
    except ImportError:
        return False

if is_hypertile_available():
    from hyper_tile import split_attention

def hypertile(unet, vae, height: int, width: int) -> ExitStack:
    s = ExitStack()
    s.enter_context(split_attention(vae, height, width, tile_size=config.api.hypertile_vae_chunk))
    s.enter_context(split_attention(unet, height, width, tile_size=config.api.hypertile_unet_chunk))
    return s