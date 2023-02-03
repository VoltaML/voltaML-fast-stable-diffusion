from typing import TYPE_CHECKING, Dict, Union

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
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

from core.types import PyTorchModelType

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion


def change_scheduler(
    model: Union[PyTorchModelType, "DemoDiffusion"],
    scheduler: KarrasDiffusionSchedulers,
    config: Dict,
):
    "Change the scheduler of the model"

    new_scheduler = None

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
    else:
        new_scheduler = model.scheduler  # type: ignore

    model.scheduler = new_scheduler.from_config(config=config)  # type: ignore
