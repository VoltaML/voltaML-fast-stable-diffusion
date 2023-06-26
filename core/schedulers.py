import importlib
import logging
from typing import Dict, Optional

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

logger = logging.getLogger(__name__)


def change_scheduler(
    model: Optional[PyTorchModelType],
    scheduler: KarrasDiffusionSchedulers,
    config: Optional[Dict] = None,
    autoload: bool = True,
    use_karras_sigmas: bool = False,
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
        if scheduler.value in [10, 11]:
            logger.debug(
                f"Loading scheduler {new_scheduler.__class__.__name__} with config karras_sigmas={use_karras_sigmas}"
            )
            model.scheduler = new_scheduler.from_config(config=config, use_karras_sigmas=use_karras_sigmas)  # type: ignore
        else:
            model.scheduler = new_scheduler.from_config(config=config)  # type: ignore
    else:
        return new_scheduler
