from contextlib import ExitStack

import torch

from core.config import config
from .autocast_utils import autocast
from .hypertile import is_hypertile_available, hypertile

class InferenceContext(ExitStack):
    """inference context"""

    unet: torch.nn.Module
    vae: torch.nn.Module


def inference_context(unet, vae, height, width) -> InferenceContext:
    "Helper function for centralizing context management"
    s = InferenceContext()
    s.unet = unet
    s.vae = vae
    s.enter_context(autocast(unet.dtype, disable=config.api.autocast))
    if is_hypertile_available() and config.api.hypertile:
        s.enter_context(hypertile(unet, height, width))
    if config.api.torch_compile:
        s.unet = torch.compile(  # type: ignore
            unet,
            fullgraph=config.api.torch_compile_fullgraph,
            dynamic=config.api.torch_compile_dynamic,
            mode=config.api.torch_compile_mode,
        )
    return s
