from typing import Callable, Optional, Union, Tuple
import functools

import torch

from .denoiser import create_denoiser
from .adapter import KdiffusionSchedulerAdapter

samplers_kdiffusion = {}

def create_sampler(
        alphas_cumprod: torch.Tensor,
        prediction_type: str,

        eta_noise_seed_delta: Optional[float] = None,
        denoiser_enable_quantization: bool = False,

        sigma_scheduler: Optional[str] = None,
        sigma_use_old_karras_scheduler: bool = False,
        sigma_always_discard_next_to_last: bool = False,
        sigma_rho: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,

        sampler_name: Optional[str] = None,
        sampler_eta: Optional[float] = None,
        sampler_churn: Optional[float] = None,
        sampler_tmin: Optional[float] = None,
        sampler_tmax: Optional[float] = None,
        sampler_noise: Optional[float] = None
):
    sampler_tuple: Union[None, Tuple[str, str]] = next((sampler for sampler in samplers_kdiffusion if sampler[0] == sampler_name), None)
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
