from typing import TYPE_CHECKING, Union

from diffusers.pipelines.stable_diffusion import StableDiffusionKDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
    DPMSolverSinglestepScheduler,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler

from core.types import KDiffusionScheduler, Scheduler

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion


def change_scheduler(
    model: Union[StableDiffusionKDiffusionPipeline, "DemoDiffusion"],
    scheduler: Union[Scheduler, KDiffusionScheduler],
    config=None,
):
    "Get the scheduler from the scheduler enum"

    if isinstance(model, StableDiffusionKDiffusionPipeline):
        model.set_scheduler(scheduler.value)
        return

    new_scheduler = None

    if scheduler == Scheduler.ddim:
        new_scheduler = DDIMScheduler
    elif scheduler == Scheduler.heun:
        new_scheduler = HeunDiscreteScheduler
    elif scheduler == Scheduler.dpm_discrete:
        new_scheduler = KDPM2DiscreteScheduler
    elif scheduler == Scheduler.dpm_ancestral:
        new_scheduler = KDPM2AncestralDiscreteScheduler
    elif scheduler == Scheduler.lms:
        new_scheduler = LMSDiscreteScheduler
    elif scheduler == Scheduler.pndm:
        new_scheduler = PNDMScheduler
    elif scheduler == Scheduler.euler:
        new_scheduler = EulerDiscreteScheduler
    elif scheduler == Scheduler.euler_a:
        new_scheduler = EulerAncestralDiscreteScheduler
    elif scheduler == Scheduler.dpmpp_sde_ancestral:
        new_scheduler = DPMSolverSinglestepScheduler
    elif scheduler == Scheduler.dpmpp_2m:
        new_scheduler = DPMSolverMultistepScheduler
    else:
        new_scheduler = model.scheduler  # type: ignore

    model.scheduler = new_scheduler.from_config(config)  # type: ignore
