import logging

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

from core.config import config

try:
    force_autocast = [torch.float8_e4m3fn, torch.float8_e5m2]
except AttributeError:
    force_autocast = []

logger = logging.getLogger(__name__)


def cast(
    pipe: StableDiffusionPipeline,
    device: str,
    dtype: torch.dtype,
    offload: bool,
):
    # Change the order of the channels to be more efficient for the GPU
    # DirectML only supports contiguous memory format
    # Disable for IPEX as well, they don't like torch's way of setting memory format
    memory_format = torch.preserve_format
    if config.api.channels_last:
        if "privateuseone" in device:
            logger.warn(
                "Optimization: Skipping channels_last, since DirectML doesn't support it."
            )
        else:
            memory_format = torch.channels_last
            if hasattr(pipe, "unet"):
                pipe.unet.to(memory_format=memory_format)
            if hasattr(pipe, "vae"):
                pipe.vae.to(memory_format=memory_format)
            logger.info("Optimization: Enabled channels_last memory format")

    pipe.unet.force_autocast = dtype in force_autocast  # type: ignore
    if pipe.unet.force_autocast:
        for b in [x for x in pipe.components.values() if hasattr(x, "modules")]:  # type: ignore
            if "CLIP" in b.__class__.__name__:
                b.to(device=None if offload else device, dtype=config.api.load_dtype)
            else:
                for module in b.modules():
                    if any(
                        [
                            x
                            for x in ["Conv", "Linear"]  # 'cause LoRACompatibleConv
                            if x in module.__class__.__name__
                        ]
                    ):
                        if hasattr(module, "fp16_weight"):
                            del module.fp16_weight
                        if config.api.cache_fp16_weight:
                            module.fp16_weight = module.weight.clone().half()
                        module.to(
                            device=None if offload else device,
                            dtype=dtype,
                        )
                    else:
                        module.to(
                            device=None if offload else device,
                            dtype=config.api.load_dtype,
                        )
        if not config.api.autocast:
            logger.info("Optimization: Forcing autocast on due to float8 weights.")
    else:
        pipe.to(
            device=None if offload else device, dtype=dtype, memory_format=memory_format
        )

    return pipe
