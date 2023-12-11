from contextlib import ExitStack
from typing import List, Optional, Type

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from core.config import config
from core.flags import AnimateDiffFlag, Flag
from .autocast_utils import autocast
from .dtype import cast
from .hypertile import is_hypertile_available, hypertile


class InferenceContext(ExitStack):
    """inference context"""

    unet: torch.nn.Module
    vae: torch.nn.Module
    flags: List[Optional[Flag]] = []
    components: dict = {}

    def to(self, device: str, dtype: torch.dtype):
        self.unet.to(device=device, dtype=dtype)
        self.vae.to(device=device, dtype=dtype)

    def get_flag(self, _type: Type) -> Optional[Flag]:
        try:
            return [x for x in self.flags if isinstance(x, _type)].pop()
        except IndexError:
            return None


def inference_context(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    height: int,
    width: int,
    flags: List[Optional[Flag]] = [],
) -> InferenceContext:
    "Helper function for centralizing context management"
    s = InferenceContext()
    s.unet = unet
    s.vae = vae
    s.components = {"unet": unet, "vae": vae}
    s.enter_context(
        autocast(
            config.api.load_dtype,
            disable=config.api.autocast and not unet.force_autocast,
        )
    )
    s.flags = flags

    animatediff: Optional[AnimateDiffFlag] = s.get_flag(AnimateDiffFlag)  # type: ignore
    if animatediff is not None:
        from core.inference.utilities.animatediff.models.unet import (
            UNet3DConditionModel,
        )
        from core.inference.utilities.animatediff import patch as patch_animatediff

        offload = s.unet.device.type == "cpu"

        s.unet = UNet3DConditionModel.from_pretrained_2d(  # type: ignore
            s.unet, animatediff.motion_model  # type: ignore
        )

        if config.api.clear_memory_policy == "always":
            from core.shared_dependent import gpu

            gpu.memory_cleanup()

        s.components.update({"unet": s.unet})
        cast(s, device=config.api.device, dtype=config.api.dtype, offload=offload)  # type: ignore
        patch_animatediff(s)
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
