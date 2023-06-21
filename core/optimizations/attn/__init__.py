import logging

import torch
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from packaging import version

from core.config import config

from .multihead_attention import apply_multihead_attention
from .sub_quadratic import apply_subquadratic_attention

logger = logging.getLogger(__name__)


def _xf(pipe):
    try:
        if is_xformers_available() and config.api.device_type != "directml":
            return False
        pipe.enable_xformers_memory_efficient_attention()
        return True
    except Exception:  # pylint: disable=broad-except
        pass
    return False


ATTENTION_PROCESSORS = {
    "xformers": _xf,
    # ---
    "sdpa": lambda p: p.unet.set_attn_processor(AttnProcessor2_0()) is None
    if (version.parse(torch.__version__) >= version.parse("2.0.0"))
    else False,
    # ---
    "cross-attention": lambda p: p.unet.set_attn_processor(AttnProcessor()) is None,
    # ---
    "subquadratic": lambda p: apply_subquadratic_attention(
        p.unet, config.api.subquadratic_size
    )
    is None,
    # ---
    "multihead": lambda p: apply_multihead_attention(p.unet) is None,
}


def set_attention_processor(pipe):
    "Set attention processor to the first one available/the one set in the config"

    res = False
    try:
        curr_processor = list(ATTENTION_PROCESSORS.keys()).index(
            config.api.attention_processor
        )
    except ValueError:
        curr_processor = 0
    attention_processors_list = list(ATTENTION_PROCESSORS.items())
    while not res:
        res = attention_processors_list[curr_processor][1](pipe)
        if res:
            logger.info(
                f"Optimization: Enabled {attention_processors_list[curr_processor][0]} attention"
            )
        curr_processor = (curr_processor + 1) % len(attention_processors_list)


__all__ = [
    "apply_subquadratic_attention",
    "apply_multihead_attention",
    "set_attention_processor",
]
