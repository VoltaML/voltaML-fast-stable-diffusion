from dataclasses import dataclass
from typing import Optional


@dataclass
class DPM_2S_a:
    eta_noise_seed_delta: Optional[float] = None
    denoiser_enable_quantization: Optional[bool] = None
    karras_sigma_scheduler: Optional[bool] = None
    sigma_use_old_karras_scheduler: Optional[bool] = None
    sigma_always_discard_next_to_last: Optional[bool] = None
    sigma_rho: Optional[float] = None
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    sampler_eta: Optional[float] = None
    sampler_churn: Optional[float] = None
    sampler_tmin: Optional[float] = None
    sampler_tmax: Optional[float] = None
    sampler_noise: Optional[float] = None
