from typing import Dict, Optional

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    UniPCMultistepScheduler,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import PyTorchModelType


def change_scheduler(
    model: Optional[PyTorchModelType],
    scheduler: KarrasDiffusionSchedulers,
    config: Optional[Dict] = None,
    autoload: bool = True,
):
    "Change the scheduler of the model"

    new_scheduler = None

    if not isinstance(
        model,
        (
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionPipeline,
            StableDiffusionUpscalePipeline,
            StableDiffusionControlNetPipeline,
        ),
    ):
        if config is None:
            raise ValueError("config must be provided for TRT model")
    else:
        config = model.scheduler.config  # type: ignore

    if scheduler == KarrasDiffusionSchedulers.DDIMScheduler:
        new_scheduler = DDIMScheduler
    elif scheduler == KarrasDiffusionSchedulers.DDPMScheduler:
        new_scheduler = DDPMScheduler
    elif scheduler == KarrasDiffusionSchedulers.DEISMultistepScheduler:
        new_scheduler = DEISMultistepScheduler
    elif scheduler == KarrasDiffusionSchedulers.HeunDiscreteScheduler:
        new_scheduler = HeunDiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.KDPM2DiscreteScheduler:
        new_scheduler = KDPM2DiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.KDPM2AncestralDiscreteScheduler:
        new_scheduler = KDPM2AncestralDiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.LMSDiscreteScheduler:
        new_scheduler = LMSDiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.PNDMScheduler:
        new_scheduler = PNDMScheduler
    elif scheduler == KarrasDiffusionSchedulers.EulerDiscreteScheduler:
        new_scheduler = EulerDiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler:
        new_scheduler = EulerAncestralDiscreteScheduler
    elif scheduler == KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler:
        new_scheduler = DPMSolverSinglestepScheduler
    elif scheduler == KarrasDiffusionSchedulers.DPMSolverMultistepScheduler:
        new_scheduler = DPMSolverMultistepScheduler
    elif scheduler == KarrasDiffusionSchedulers.UniPCMultistepScheduler:
        new_scheduler = UniPCMultistepScheduler
    else:
        new_scheduler = model.scheduler  # type: ignore

    if autoload:
        model.scheduler = new_scheduler.from_config(config=config)  # type: ignore
    else:
        return new_scheduler
