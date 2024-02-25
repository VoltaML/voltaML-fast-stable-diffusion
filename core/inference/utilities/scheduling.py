import importlib
import inspect
import logging
from typing import Dict, Optional, Union

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)

from core import shared
from core.config import config
from core.inference.utilities.philox import PhiloxGenerator
from core.scheduling import KdiffusionSchedulerAdapter, create_sampler
from core.scheduling.custom.sasolver import SASolverScheduler
from core.types import PyTorchModelType, SigmaScheduler
from core.utils import unwrap_enum

logger = logging.getLogger(__name__)


def get_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
    is_text2img: bool,
):
    "Get the amount of timesteps for the provided options"

    shared.current_done_steps = 0

    if is_text2img:
        shared.current_steps = num_inference_steps
        return scheduler.timesteps.to(device), num_inference_steps  # type: ignore
    else:
        # get the original timestep using init_timestep
        offset = scheduler.config.get("steps_offset", 0)  # type: ignore
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:].to(device)  # type: ignore

        shared.current_steps = num_inference_steps - t_start
        return timesteps, num_inference_steps - t_start


def prepare_extra_step_kwargs(
    scheduler: SchedulerMixin,
    eta: Optional[float],
    generator: Union[PhiloxGenerator, torch.Generator],
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
    accepts_generator = (
        "generator" in set(inspect.signature(scheduler.step).parameters.keys())  # type: ignore
        and config.api.generator != "philox"
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def change_scheduler(
    model: Optional[PyTorchModelType],
    scheduler: Union[str, KarrasDiffusionSchedulers],
    configuration: Optional[Dict] = None,
    sigma_type: SigmaScheduler = "automatic",
    sampler_settings: Optional[Dict] = None,
) -> SchedulerMixin:
    "Change the scheduler of the model"

    configuration = model.scheduler.config  # type: ignore

    if (isinstance(scheduler, str) and scheduler.isdigit()) or isinstance(
        scheduler, (int, KarrasDiffusionSchedulers)
    ):
        scheduler = KarrasDiffusionSchedulers(int(unwrap_enum(scheduler)))
        try:
            new_scheduler = getattr(
                importlib.import_module("diffusers"), scheduler.name
            )
        except AttributeError:
            new_scheduler = model.scheduler  # type: ignore

        if scheduler.value in [10, 11]:
            logger.debug(
                f"Loading scheduler {new_scheduler.__class__.__name__} with config sigmas={sigma_type}"
            )
            new_scheduler = new_scheduler.from_config(
                config=configuration, use_karras_sigmas=sigma_type == ""
            )  # type: ignore
        else:
            new_scheduler = new_scheduler.from_config(config=configuration)  # type: ignore
    else:
        sched = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )  # type: ignore

        if scheduler == "sasolver":
            new_scheduler = SASolverScheduler.from_config(config=configuration)  # type: ignore
        else:
            new_scheduler = create_sampler(
                alphas_cumprod=sched.alphas_cumprod,  # type: ignore
                denoiser_enable_quantization=config.api.kdiffusers_quantization,
                sampler=scheduler,
                sigma_type=sigma_type,
                eta_noise_seed_delta=0,
                sigma_always_discard_next_to_last=False,
                sigma_use_old_karras_scheduler=False,
                device=torch.device(config.api.device),  # type: ignore
                dtype=config.api.load_dtype,  # type: ignore
                sampler_settings=sampler_settings,
            )
    model.scheduler = new_scheduler  # type: ignore
    return new_scheduler  # type: ignore
