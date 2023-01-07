from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
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

from core.types import Scheduler


def change_scheduler(model: StableDiffusionPipeline, scheduler: Scheduler):
    "Get the scheduler from the scheduler enum"

    config = model.scheduler.config  # type: ignore
    new_scheduler = None

    if scheduler == scheduler.ddim:
        new_scheduler = DDIMScheduler
    elif scheduler == scheduler.heun:
        new_scheduler = HeunDiscreteScheduler
    elif scheduler == scheduler.dpm_discrete:
        new_scheduler = KDPM2DiscreteScheduler
    elif scheduler == scheduler.dpm_ancestral:
        new_scheduler = KDPM2AncestralDiscreteScheduler
    elif scheduler == scheduler.lms:
        new_scheduler = LMSDiscreteScheduler
    elif scheduler == scheduler.pndm:
        new_scheduler = PNDMScheduler
    elif scheduler == scheduler.euler:
        new_scheduler = EulerDiscreteScheduler
    elif scheduler == scheduler.euler_a:
        new_scheduler = EulerAncestralDiscreteScheduler
    elif scheduler == scheduler.dpmpp_sde_ancestral:
        new_scheduler = DPMSolverSinglestepScheduler
    elif scheduler == scheduler.dpmpp_2m:
        new_scheduler = DPMSolverMultistepScheduler
    else:
        new_scheduler = model.scheduler  # type: ignore

    model.scheduler = new_scheduler.from_config(config)  # type: ignore
