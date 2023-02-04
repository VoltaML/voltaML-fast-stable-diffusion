from typing import TYPE_CHECKING, Dict, Optional, Union

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_depth2img import (
    StableDiffusionDepth2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_image_variation import (
    StableDiffusionImageVariationPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
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
    config: Optional[Dict] = None,
):
    "Change the scheduler of the model"

    new_scheduler = None

    if not isinstance(
        model,
        (
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionPipeline,
            StableDiffusionUpscalePipeline,
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
    else:
        new_scheduler = model.scheduler  # type: ignore

    model.scheduler = new_scheduler.from_config(config=config)  # type: ignore
