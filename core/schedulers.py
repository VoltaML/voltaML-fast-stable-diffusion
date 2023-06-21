from typing import Dict, Optional
import importlib

from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
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

    if not isinstance(
        model,
        (
            DiffusionPipeline,
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

    try:
        new_scheduler = getattr(importlib.import_module("diffusers"), scheduler.name)
    except AttributeError:
        new_scheduler = model.scheduler  # type: ignore

    if autoload:
        model.scheduler = new_scheduler.from_config(config=config)  # type: ignore
    else:
        return new_scheduler
