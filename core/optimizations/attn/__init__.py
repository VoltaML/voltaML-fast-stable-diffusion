from packaging import version

from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
import torch

from core.config import config
from .sub_quadratic import apply_subquadratic_attention
from .multihead_attention import apply_multihead_attention


ATTENTION_PROCESSORS = {
    "xformers": lambda p: p.enable_xformers_memory_efficient_attention() is None
    if (is_xformers_available() and config.api.device_type != "directml")
    else False,
    # ---
    "sdpa": lambda p: p.unet.set_attn_processor(AttnProcessor2_0()) is None
    if (version.parse(torch.__version__) >= version.parse("2.0.0"))
    else False,
    # ---
    "cross-attention": lambda p: p.unet.set_attn_processor(AttnProcessor()) is None,
    # ---
    "subquadratic": lambda p: apply_subquadratic_attention(p.unet) is None,
    # ---
    "multihead": lambda p: apply_multihead_attention(p.unet) is None,
}


def set_attention_processor(pipe, logger):
    "Set attention processor to the first one available/the one set in the config"

    res = False
    try:
        curr_processor = list(ATTENTION_PROCESSORS.keys()).index(
            config.api.attention_processor
        )
    except ValueError:
        curr_processor = 0
    l = list(ATTENTION_PROCESSORS.items())
    while not res:
        res = l[curr_processor][1](pipe)
        if res:
            logger.info(f"Optimization: Enabled {l[curr_processor][0]} attention")
        curr_processor = (curr_processor + 1) % len(l)


__all__ = [
    "apply_subquadratic_attention",
    "apply_multihead_attention",
    "set_attention_processor",
]
