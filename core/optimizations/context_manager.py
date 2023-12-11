from contextlib import ExitStack
from typing import List, Optional

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unet_motion_model import UNetMotionModel, MotionAdapter

from core.config import config
from core.flags import AnimateDiffFlag, Flag
from .autocast_utils import autocast
from .dtype import cast
from .hypertile import is_hypertile_available, hypertile


class InferenceContext(ExitStack):
    """inference context"""

    unet: torch.nn.Module
    vae: torch.nn.Module
    components: dict = {}

    def to(self, device: str, dtype: torch.dtype):
        self.unet.to(device=device, dtype=dtype)
        self.vae.to(device=device, dtype=dtype)


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
    animatediff = [x for x in flags if isinstance(x, AnimateDiffFlag)]
    print(len(flags) - len(animatediff))
    if len(animatediff) != 0:
        flag: AnimateDiffFlag = animatediff[0]
        motion_adapter: MotionAdapter = MotionAdapter.from_pretrained(flag.motion_model)  # type: ignore
        motion_adapter.to(dtype=config.api.load_dtype, device=config.api.device)

        offload = s.unet.device.type == "cpu"

        s.unet = UNetMotionModel.from_unet2d(  # type: ignore
            s.unet, motion_adapter  # type: ignore
        )

        if flag.chunk_feed_forward:
            s.unet.enable_forward_chunking()  # type: ignore

        s.components.update({"unet": s.unet})
        cast(s, device=config.api.device, dtype=config.api.dtype, offload=offload)  # type: ignore

        s.unet.to(dtype=config.api.load_dtype, device=config.api.device)  # type: ignore
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
