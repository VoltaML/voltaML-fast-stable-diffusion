from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseMixin:
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    sigma_rho: Optional[float] = None
    sigma_discard: Optional[bool] = None
    sampler_tmin: Optional[float] = None
    sampler_tmax: Optional[float] = None
    sampler_noise: Optional[float] = None


@dataclass
class AncestralMixin:
    eta: Optional[float] = None


@dataclass
class Euler(BaseMixin):
    s_churn: Optional[float] = None


@dataclass
class Euler_a(BaseMixin, AncestralMixin):
    noise_sampler: Optional[str] = None


@dataclass
class Heun(BaseMixin):
    s_churn: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class Heunpp(BaseMixin):
    s_churn: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class DPM_2(BaseMixin):
    s_churn: Optional[float] = None
    s_noise: Optional[float] = None


@dataclass
class DPM_2_a(BaseMixin, AncestralMixin):
    pass


@dataclass
class LMS(BaseMixin):
    order: Optional[int] = None


@dataclass
class DPM_fast(BaseMixin, AncestralMixin):
    pass


@dataclass
class DPM_adaptive(BaseMixin, AncestralMixin):
    order: Optional[int] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    h_init: Optional[float] = None
    pcoeff: Optional[float] = None
    icoeff: Optional[float] = None
    dcoeff: Optional[float] = None
    accept_safety: Optional[float] = None


@dataclass
class DPMpp_2S_a(BaseMixin, AncestralMixin):
    pass


@dataclass
class DPMpp_SDE(BaseMixin, AncestralMixin):
    r: Optional[float] = None


@dataclass
class DPMpp_2M(BaseMixin):
    pass


@dataclass
class DPMpp_2M_SDE(BaseMixin, AncestralMixin):
    solver_type: Optional[str] = None


@dataclass
class DPMpp_3M_SDE(BaseMixin, AncestralMixin):
    pass
