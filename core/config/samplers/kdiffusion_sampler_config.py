from dataclasses import dataclass
from typing import Optional


@dataclass
class AncestralMixin:
    eta: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class Euler:
    s_churn: Optional[float] = None
    s_tmin: Optional[float] = None
    s_tmax: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class Euler_a(AncestralMixin):
    noise_sampler: Optional[str] = None


@dataclass
class Heun:
    s_churn: Optional[float] = None
    s_tmin: Optional[float] = None
    s_tmax: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class DPM_2:
    s_churn: Optional[float] = None
    s_tmin: Optional[float] = None
    s_tmax: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class DPM_2_a(AncestralMixin):
    noise_sampler: Optional[str] = None


@dataclass
class LMS:
    order: Optional[int] = None


@dataclass
class DPM_fast(AncestralMixin):
    noise_sampler: Optional[str] = None


@dataclass
class DPM_adaptive(AncestralMixin):
    order: Optional[int] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    h_init: Optional[float] = None
    pcoeff: Optional[float] = None
    icoeff: Optional[float] = None
    dcoeff: Optional[float] = None
    accept_safety: Optional[float] = None
    noise_sampler: Optional[str] = None


@dataclass
class DPMpp_2S_a(AncestralMixin):
    noise_sampler: Optional[str] = None


@dataclass
class DPMpp_SDE(AncestralMixin):
    noise_sampler: Optional[str] = None
    r: Optional[float] = None


@dataclass
class DPMpp_2M:
    pass


@dataclass
class DPMpp_2M_SDE(AncestralMixin):
    noise_sampler: Optional[str] = None
    solver_type: Optional[str] = None


@dataclass
class DPMpp_3M_SDE(AncestralMixin):
    noise_sampler: Optional[str] = None
