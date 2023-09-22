from typing import Optional, Union, Tuple

import torch
import kdiffusion

from .denoiser import create_denoiser, Denoiser
from .sigmas import build_sigmas

sampling = kdiffusion.sampling
samplers_kdiffusion = {}

def create_sampler(
        unet: torch.Module,
        alphas_cumprod,
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
    sampler_func = getattr(sampling, sampler_tuple[1])
    sampler_info = getattr(sampling, sampler_tuple[2])
