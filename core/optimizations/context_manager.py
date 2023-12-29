from contextlib import ExitStack
from types import TracebackType
from typing import List, Optional, Type

import torch
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from core.config import config
from core.flags import AnimateDiffFlag, Flag
from .autocast_utils import autocast
from .hypertile import is_hypertile_available, hypertile


class InferenceContext(ExitStack):
    """inference context"""

    old_device: torch.device
    old_dtype: torch.dtype
    old_unet: torch.nn.Module | None = None
    unet: torch.nn.Module
    vae: torch.nn.Module
    profiler: Optional[torch.profiler.profile] = None
    flags: List[Optional[Flag]] = []
    components: dict = {}

    def to(self, device: str, dtype: torch.dtype, memory_format):
        self.vae.to(device=device, dtype=dtype, memory_format=memory_format)  # type: ignore
        self.unet.to(device=device, dtype=dtype, memory_format=memory_format)  # type: ignore

    def enable_freeu(self, s1, s2, b1, b2):
        if hasattr(self.unet, "enable_freeu"):
            self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def get_flag(self, _type: Type) -> Optional[Flag]:
        try:
            return [x for x in self.flags if isinstance(x, _type)].pop()
        except IndexError:
            return None

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool:
        ret = super().__exit__(__exc_type, __exc_value, __traceback)
        if self.old_unet is not None:
            self.old_unet.to(device=self.old_device, dtype=self.old_dtype)
        return ret

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()


PROFILE = False


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
    if PROFILE:
        s.profiler = s.enter_context(
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
        )
    s.flags = flags

    s.old_device = s.unet.device
    s.old_dtype = s.unet.dtype

    animatediff: Optional[AnimateDiffFlag] = s.get_flag(AnimateDiffFlag)  # type: ignore
    if animatediff is not None:
        from core.inference.utilities.animatediff.models.unet import (
            UNet3DConditionModel,
        )
        from core.inference.utilities.animatediff import patch as patch_animatediff

        s.unet.to("cpu")
        s.old_unet = s.unet
        s.unet = UNet3DConditionModel.from_pretrained_2d(  # type: ignore
            s.unet, animatediff.motion_model  # type: ignore
        )

        if animatediff.use_pia:
            s.unet = s.unet.convert_to_pia(animatediff.pia_checkpont)

        if config.api.clear_memory_policy == "always":
            from core.shared_dependent import gpu

            gpu.memory_cleanup()

        from .pytorch_optimizations import optimize_model

        optimize_model(s, config.api.device, silent=True)  # type: ignore

        if animatediff.chunk_feed_forward != -1:
            # from core.inference.utilities.animatediff import memory_required

            # TODO: do auto batch calculation
            # for now, "auto" is 1.
            batch_size = (
                1
                if animatediff.chunk_feed_size == "auto"
                else animatediff.chunk_feed_size
            )
            s.unet.enable_forward_chunking(
                chunk_size=batch_size, dim=animatediff.chunk_feed_forward
            )
        # set_offload(s.unet, config.api.device, "model")  # type: ignore

        patch_animatediff(s)
        s.components.update({"unet": s.unet})
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
