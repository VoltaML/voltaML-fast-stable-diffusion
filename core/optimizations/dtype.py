import logging
from typing import Union

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

from core.config import config

try:
    force_autocast = [torch.float8_e4m3fn, torch.float8_e5m2]
except AttributeError:
    force_autocast = []

logger = logging.getLogger(__name__)


def cast(
    pipe: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    device: str,
    dtype: torch.dtype,
    offload: bool,
):
    # Change the order of the channels to be more efficient for the GPU
    # DirectML only supports contiguous memory format
    # Disable for IPEX as well, they don't like torch's way of setting memory format
    if config.api.channels_last:
        if "privateuseone" in device:
            logger.warn(
                "Optimization: Skipping channels_last, since DirectML doesn't support it."
            )
        else:
            if hasattr(pipe, "unet"):
                pipe.unet.to(memory_format=torch.channels_last)  # type: ignore
            if hasattr(pipe, "vae"):
                pipe.vae.to(memory_format=torch.channels_last)  # type: ignore
            logger.info("Optimization: Enabled channels_last memory format")

    pipe.unet.force_autocast = dtype in force_autocast
    if pipe.unet.force_autocast:
        for m in [x.modules() for x in pipe.components.values() if hasattr(x, "modules")]:  # type: ignore
            if "CLIP" in m.__class__.__name__:
                m.to(device=None if offload else device, dtype=config.api.load_dtype)
            else:
                for module in m:
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
                        module.to(device=None if offload else device, dtype=dtype)
                    else:
                        module.to(
                            device=None if offload else device,
                            dtype=config.api.load_dtype,
                        )
        if not config.api.autocast:
            logger.info("Optimization: Forcing autocast on due to float8 weights.")
    else:
        pipe.to(device=None if offload else device, dtype=dtype)

    return pipe
