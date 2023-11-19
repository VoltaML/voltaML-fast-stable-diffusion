import logging
from typing import Callable, Optional, Tuple, Union

import torch
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from core.types import SigmaScheduler

from .adapter.k_adapter import KdiffusionSchedulerAdapter
from .adapter.unipc_adapter import UnipcSchedulerAdapter
from .custom.dpmpp_2m import sample_dpmpp_2mV2
from .custom.restart import restart_sampler
from .custom.lcm import sample_lcm
from .denoiser import create_denoiser

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
    ("euler_a", "sample_euler_ancestral", {"uses_ensd": True}),
    ("euler", "sample_euler", {}),
    ("lms", "sample_lms", {}),
    ("heun", "sample_heun", {"second_order": True}),
    ("dpm_fast", "sample_dpm_fast", {"uses_ensd": True, "default_eta": 0.0}),
    ("dpm_adaptive", "sample_dpm_adaptive", {"uses_ensd": True, "default_eta": 0.0}),
    (
        "dpm2",
        "sample_dpm_2",
        {
            "discard_next_to_last_sigma": True,
            "uses_ensd": True,
            "second_order": True,
        },
    ),
    (
        "dpm2_a",
        "sample_dpm_2_ancestral",
        {
            "discard_next_to_last_sigma": True,
            "uses_ensd": True,
            "second_order": True,
        },
    ),
    (
        "dpmpp_2s_a",
        "sample_dpmpp_2s_ancestral",
        {"uses_ensd": True, "second_order": True},
    ),
    ("dpmpp_2m", "sample_dpmpp_2m", {}),
    (
        "dpmpp_sde",
        "sample_dpmpp_sde",
        {"second_order": True, "brownian_noise": True},
    ),
    (
        "dpmpp_2m_sharp",
        sample_dpmpp_2mV2,  # pretty much experimental, only for testing things
        {},
    ),
    (
        "dpmpp_2m_sde",
        "sample_dpmpp_2m_sde",
        {"brownian_noise": True},
    ),
    (
        "dpmpp_3m_sde",
        "sample_dpmpp_3m_sde",
        {"brownian_noise": True},
    ),
    ("unipc_multistep", "unipc", {}),
    ("restart", sample_lcm, {}),  # restart_sampler
    ("lcm", sample_lcm, {}),
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
    sigma_type: SigmaScheduler = "automatic",
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
    sampler_settings: Optional[dict] = None,
):
    "Helper function for figuring out and creating a KdiffusionSchedulerAdapter for the appropriate settings given."
    sampler_tuple = _get_sampler(sampler)
    sampler_settings = sampler_settings or {}

    sigma_min = sampler_settings.pop("sigma_min", sigma_min)
    sigma_max = sampler_settings.pop("sigma_max", sigma_max)
    sigma_rho = sampler_settings.pop("sigma_rho", sigma_rho)
    sigma_always_discard_next_to_last = sampler_settings.pop(
        "sigma_discard", sigma_always_discard_next_to_last
    )
    sampler_tmin = sampler_settings.pop("sampler_tmin", sampler_tmin)
    sampler_tmax = sampler_settings.pop("sampler_tmax", sampler_tmax)
    sampler_noise = sampler_settings.pop("sampler_noise", sampler_noise)

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
        scheduler_name = sampler_tuple[2].get("scheduler", sigma_type)
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
            sampler_settings=sampler_settings,
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
