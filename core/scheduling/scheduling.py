from typing import Callable, Optional, Union, Tuple
import logging

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
import torch

from .custom.dpmpp_2m import sample_dpmpp_2mV2
from .custom.restart import restart_sampler
from .denoiser import create_denoiser
from .adapter.k_adapter import KdiffusionSchedulerAdapter
from .adapter.unipc_adapter import UnipcSchedulerAdapter

logger = logging.getLogger(__name__)

samplers_diffusers = {
    KarrasDiffusionSchedulers.DPMSolverMultistepScheduler: "DPM++ 2M Karras",
    KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler: "DPM++ 2M SDE Karras",
    KarrasDiffusionSchedulers.KDPM2DiscreteScheduler: "DPM2 Karras",
    KarrasDiffusionSchedulers.KDPM2AncestralDiscreteScheduler: "DPM2 a Karras",
    KarrasDiffusionSchedulers.EulerDiscreteScheduler: "Euler",
    KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler: "Euler a",
    KarrasDiffusionSchedulers.HeunDiscreteScheduler: "Heun",
    KarrasDiffusionSchedulers.LMSDiscreteScheduler: "LMS Karras",
    KarrasDiffusionSchedulers.UniPCMultistepScheduler: "UniPC Multistep",
}
samplers_kdiffusion = [
    ("Euler a", "sample_euler_ancestral", {"uses_ensd": True}),
    ("Euler", "sample_euler", {}),
    ("LMS", "sample_lms", {}),
    ("Heun", "sample_heun", {"second_order": True}),
    ("DPM fast", "sample_dpm_fast", {"uses_ensd": True, "default_eta": 0.0}),
    ("DPM adaptive", "sample_dpm_adaptive", {"uses_ensd": True, "default_eta": 0.0}),
    (
        "DPM2",
        "sample_dpm_2",
        {
            "discard_next_to_last_sigma": True,
            "uses_ensd": True,
            "second_order": True,
        },
    ),
    (
        "DPM2 a",
        "sample_dpm_2_ancestral",
        {
            "discard_next_to_last_sigma": True,
            "uses_ensd": True,
            "second_order": True,
        },
    ),
    (
        "DPM++ 2S a",
        "sample_dpmpp_2s_ancestral",
        {"uses_ensd": True, "second_order": True},
    ),
    ("DPM++ 2M", "sample_dpmpp_2m", {}),
    (
        "DPM++ SDE",
        "sample_dpmpp_sde",
        {"second_order": True, "brownian_noise": True},
    ),
    (
        "DPM++ 2M Sharp",
        sample_dpmpp_2mV2,  # pretty much experimental, only for testing things
        {},
    ),
    (
        "DPM++ 2M SDE",
        "sample_dpmpp_2m_sde",
        {"brownian_noise": True},
    ),
    (
        "DPM++ 3M SDE",
        "sample_dpmpp_3m_sde",
        {"brownian_noise": True},
    ),
    ("UniPC Multistep", "unipc", {}),
    ("Restart", restart_sampler, {}),
]


def _get_sampler(
    sampler: Union[str, KarrasDiffusionSchedulers]
) -> Union[None, Tuple[str, Union[Callable, str], dict]]:
    if isinstance(sampler, KarrasDiffusionSchedulers):
        sampler = samplers_diffusers.get(sampler, "Euler a")  # type: ignore
    return next(
        (ksampler for ksampler in samplers_kdiffusion if ksampler[0] == sampler), None
    )


def create_sampler(
    alphas_cumprod: torch.Tensor,
    sampler: Union[str, KarrasDiffusionSchedulers],
    device: torch.device,
    dtype: torch.dtype,
    eta_noise_seed_delta: Optional[float] = None,
    denoiser_enable_quantization: bool = False,
    karras_sigma_scheduler: bool = False,
    sigma_use_old_karras_scheduler: bool = False,
    sigma_always_discard_next_to_last: bool = False,
    sigma_rho: Optional[float] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    sampler_eta: Optional[float] = None,
    sampler_churn: Optional[float] = None,
    sampler_tmin: Optional[float] = None,
    sampler_tmax: Optional[float] = None,
    sampler_noise: Optional[float] = None,
):
    "Helper function for figuring out and creating a KdiffusionSchedulerAdapter for the appropriate settings given."
    sampler_tuple = _get_sampler(sampler)
    if sampler_tuple is None:
        raise ValueError("sampler_tuple is invalid")

    if sampler_tuple[1] == "unipc":
        adapter = UnipcSchedulerAdapter(
            alphas_cumprod=alphas_cumprod,
            device=device,
            dtype=dtype,
            **sampler_tuple[2],
        )
    else:
        scheduler_name = sampler_tuple[2].get(
            "scheduler", "karras" if karras_sigma_scheduler else None
        )
        if scheduler_name == "karras" and sigma_use_old_karras_scheduler:
            sigma_min = 0.1
            sigma_max = 10

        prediction_type = sampler_tuple[2].get("prediction_type", "epsilon")
        logger.debug(f"Selected scheduler: {sampler_tuple[0]}-{prediction_type}")

        adapter = KdiffusionSchedulerAdapter(
            alphas_cumprod=alphas_cumprod,
            scheduler_name=scheduler_name,
            sampler_tuple=sampler_tuple,
            sigma_range=(sigma_min, sigma_max),  # type: ignore
            sigma_rho=sigma_rho,  # type: ignore
            sigma_discard=sigma_always_discard_next_to_last,
            sampler_churn=sampler_churn,  # type: ignore
            sampler_eta=sampler_eta,  # type: ignore
            sampler_noise=sampler_noise,  # type: ignore
            sampler_trange=(sampler_tmin, sampler_tmax),  # type: ignore
            device=device,
            dtype=dtype,
        )

        adapter.eta_noise_seed_delta = eta_noise_seed_delta or 0

        adapter.denoiser = create_denoiser(
            alphas_cumprod=alphas_cumprod,
            prediction_type=prediction_type,
            denoiser_enable_quantization=denoiser_enable_quantization,
            device=device,
            dtype=dtype,
        )
    return adapter
