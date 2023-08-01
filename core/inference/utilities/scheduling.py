import importlib
import inspect
import logging
from typing import Dict, Optional

import torch
from diffusers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import PyTorchModelType

logger = logging.getLogger(__name__)


def get_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
    is_text2img: bool,
):
    "Get the amount of timesteps for the provided options"
    if is_text2img:
        return scheduler.timesteps.to(device), num_inference_steps  # type: ignore
    else:
        # get the original timestep using init_timestep
        offset = scheduler.config.get("steps_offset", 0)  # type: ignore
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:].to(device)  # type: ignore
        return timesteps, num_inference_steps - t_start


def prepare_extra_step_kwargs(
    scheduler: SchedulerMixin, generator: torch.Generator, eta: float
):
    """prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    and should be between [0, 1]"""

    accepts_eta = "eta" in set(
        inspect.signature(scheduler.step).parameters.keys()  # type: ignore
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(scheduler.step).parameters.keys()  # type: ignore
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def change_scheduler(
    model: Optional[PyTorchModelType],
    scheduler: KarrasDiffusionSchedulers,
    config: Optional[Dict] = None,
    autoload: bool = True,
    use_karras_sigmas: bool = False,
):
    "Change the scheduler of the model"

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
