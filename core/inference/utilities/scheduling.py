import importlib
import inspect
import logging
from typing import Dict, Optional, Union

import torch
from diffusers import SchedulerMixin, DDIMScheduler
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import PyTorchModelType
from core.scheduling import KdiffusionSchedulerAdapter

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

    if isinstance(scheduler, KdiffusionSchedulerAdapter):
        return {}

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
    scheduler: Union[str, KarrasDiffusionSchedulers],
    config: Optional[Dict] = None,
    use_karras_sigmas: bool = False,
):
    "Change the scheduler of the model"

    config = model.scheduler.config  # type: ignore

    if scheduler.isdigit():  # type: ignore
        scheduler = KarrasDiffusionSchedulers(int(scheduler))  # type: ignore
        try:
            new_scheduler = getattr(importlib.import_module("diffusers"), scheduler.name)  # type: ignore
        except AttributeError:
            new_scheduler = model.scheduler  # type: ignore

        if scheduler.value in [10, 11]:  # type: ignore
            logger.debug(
                f"Loading scheduler {new_scheduler.__class__.__name__} with config karras_sigmas={use_karras_sigmas}"
            )
            new_scheduler = new_scheduler.from_config(config=config, use_karras_sigmas=use_karras_sigmas)  # type: ignore
        else:
            new_scheduler = new_scheduler.from_config(config=config)  # type: ignore
    else:
        from ...scheduling import scheduling

        sched = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")  # type: ignore

        new_scheduler = scheduling.create_sampler(
            alphas_cumprod=sched.alphas_cumprod,  # type: ignore
            prediction_type=sched.config["prediction_type"],  # type: ignore
            denoiser_enable_quantization=True,
            sampler=scheduler,
            eta_noise_seed_delta=0,
            sigma_always_discard_next_to_last=False,
            sigma_use_old_karras_scheduler=False,
            device=model.unet.device,  # type: ignore
            dtype=model.unet.dtype,  # type: ignore
        )
    model.scheduler = new_scheduler  # type: ignore
    return new_scheduler
