from typing import Optional, Union, Tuple

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
import torch

from .denoiser import create_denoiser
from .adapter import KdiffusionSchedulerAdapter

samplers_diffusers = {
    KarrasDiffusionSchedulers.DPMSolverMultistepScheduler: "DPM++ 2M Karras",
    KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler: "DPM++ 2M SDE Karras",
    KarrasDiffusionSchedulers.KDPM2DiscreteScheduler: "DPM2 Karras",
    KarrasDiffusionSchedulers.KDPM2AncestralDiscreteScheduler: "DPM2 a Karras",
    KarrasDiffusionSchedulers.EulerDiscreteScheduler: "Euler",
    KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler: "Euler a",
    KarrasDiffusionSchedulers.HeunDiscreteScheduler: "Heun",
    KarrasDiffusionSchedulers.LMSDiscreteScheduler: "LMS Karras"
}
samplers_kdiffusion = [
    ("Euler a", "sample_euler_ancestral", {"uses_ensd": True}),
    ("Euler", "sample_euler", {}),
    ("LMS", "sample_lms", {}),
    ("Heun", "sample_heun", {"second_order": True}),
    ("DPM2", "sample_dpm_2", {"discard_next_to_last_sigma": True}),
    ("DPM2 a", "sample_dpm_2_ancestral", {"discard_next_to_last_sigma": True, "uses_ensd": True}),
    ("DPM++ 2S a", "sample_dpmpp_2s_ancestral", {"uses_ensd": True, "second_order": True}),
    ("DPM++ 2M", "sample_dpmpp_2m", {}),
    ("DPM++ SDE", "sample_dpmpp_sde", {"second_order": True, "brownian_noise": True}),
    ("DPM++ 2M SDE", "sample_dpmpp_2m_sde", {"brownian_noise": True}),
    ("DPM fast", "sample_dpm_fast", {"uses_ensd": True, "default_eta": 0.0}),
    ("DPM adaptive", "sample_dpm_adaptive", {"uses_ensd": True, "default_eta": 0.0}),
    ("LMS Karras", "sample_lms", {"scheduler": "karras"}),
    ("DPM2 Karras", "sample_dpm_2", {"scheduler": "karras", "discard_next_to_last_sigma": True, "uses_ensd": True, "second_order": True}),
    ("DPM2 a Karras", "sample_dpm_2_ancestral", {"scheduler": "karras", "discard_next_to_last_sigma": True, "uses_ensd": True, "second_order": True}),
    ("DPM++ 2S a Karras", "sample_dpmpp_2s_ancestral", {"scheduler": "karras", "uses_ensd": True, "second_order": True}),
    ("DPM++ 2M Karras", "sample_dpmpp_2m", {"scheduler": "karras"}),
    ("DPM++ SDE Karras", "sample_dpmpp_sde", {"scheduler": "karras", "second_order": True, "brownian_noise": True}),
    ("DPM++ 2M SDE Karras", "sample_dpmpp_2m_sde", {"scheduler": "karras", "brownian_noise": True}),
]

def _get_sampler(sampler: Union[str, KarrasDiffusionSchedulers]) -> Union[None, Tuple[str, str, dict]]:
    if sampler is not str:
        sampler = samplers_diffusers.get(sampler, "Euler a")  # type: ignore
    return next((sampler for sampler in samplers_kdiffusion if sampler[0] == sampler), None)

def create_sampler(
        alphas_cumprod: torch.Tensor,
        prediction_type: str,

        sampler: Union[str, KarrasDiffusionSchedulers],

        eta_noise_seed_delta: Optional[float] = None,
        denoiser_enable_quantization: bool = False,

        sigma_scheduler: Optional[str] = None,
        sigma_use_old_karras_scheduler: bool = False,
        sigma_always_discard_next_to_last: bool = False,
        sigma_rho: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,

        sampler_eta: Optional[float] = None,
        sampler_churn: Optional[float] = None,
        sampler_tmin: Optional[float] = None,
        sampler_tmax: Optional[float] = None,
        sampler_noise: Optional[float] = None
):
    sampler_tuple = _get_sampler(sampler)
    if sampler_tuple is None:
        raise ValueError("sampler_tuple is invalid")

    scheduler_name = sampler_tuple[2].get("scheduler", sigma_scheduler)
    if scheduler_name == "karras" and sigma_use_old_karras_scheduler:
        sigma_min = 0.1
        sigma_max = 10

    adapter = KdiffusionSchedulerAdapter(
        alphas_cumprod=alphas_cumprod,
        scheduler_name=scheduler_name,
        sampler_tuple=sampler_tuple,

        sigma_range=(sigma_min, sigma_max),
        sigma_rho=sigma_rho,
        sigma_discard=sigma_always_discard_next_to_last,

        sampler_churn=sampler_churn,
        sampler_eta=sampler_eta,
        sampler_noise=sampler_noise,
        sampler_tmax=sampler_tmax,
        sampler_tmin=sampler_tmin
    )

    adapter.eta_noise_seed_delta = eta_noise_seed_delta or 0

    adapter.denoiser = create_denoiser(
        alphas_cumprod=alphas_cumprod,
        prediction_type=prediction_type,
        denoiser_enable_quantization=denoiser_enable_quantization,
    )
