from dataclasses import dataclass, field
from typing import List, Literal, Union

from .kdiffusion_sampler_config import (
    DPM_2,
    LMS,
    DPM_2_a,
    DPM_adaptive,
    DPM_fast,
    DPMpp_2M,
    DPMpp_2M_SDE,
    DPMpp_2S_a,
    DPMpp_3M_SDE,
    DPMpp_SDE,
    Euler,
    Euler_a,
    Heun,
)


@dataclass
class Slider:
    min: int
    max: int
    step: float
    componentType: Literal["slider"] = "slider"


@dataclass
class SelectOption:
    label: str
    value: str


@dataclass
class Select:
    options: List[SelectOption]
    componentType: Literal["select"] = "select"


@dataclass
class Checkbox:
    componentType: Literal["boolean"] = "boolean"


@dataclass
class NumberInput:
    min: int
    max: int
    step: float
    componentType: Literal["number"] = "number"


FrontendComponent = Union[Slider, Select, Checkbox, NumberInput]


@dataclass
class ParamSettings:
    # K-diffusion
    eta_noise_seed_delta: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=999_999_999_999, step=1)
    )
    denoiser_enable_quantization: FrontendComponent = field(default_factory=Checkbox)
    karras_sigma_scheduler: FrontendComponent = field(default_factory=Checkbox)
    sigma_use_old_karras_scheduler: FrontendComponent = field(default_factory=Checkbox)
    sigma_discard: FrontendComponent = field(default_factory=Checkbox)
    sigma_rho: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    sigma_min: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    sigma_max: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    eta: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    s_churn: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    sampler_tmin: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    sampler_tmax: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    sampler_noise: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    noise_sampler: FrontendComponent = field(default_factory=lambda: Select([]))

    order: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=1)
    )
    rtol: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    atol: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    h_init: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    pcoeff: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    icoeff: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    dcoeff: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    accept_safety: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    r: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=10, step=0.01)
    )
    solver_type: FrontendComponent = field(default_factory=lambda: Select([]))


@dataclass
class SamplerConfig:
    # Settings for UI
    ui_settings: ParamSettings = field(default_factory=ParamSettings)

    # Samplers
    euler_a: Euler_a = field(default_factory=Euler_a)
    euler: Euler = field(default_factory=Euler)
    lms: LMS = field(default_factory=LMS)
    heun: Heun = field(default_factory=Heun)
    dpm_fast: DPM_fast = field(default_factory=DPM_fast)
    dpm_adaptive: DPM_adaptive = field(default_factory=DPM_adaptive)
    dpm2: DPM_2 = field(default_factory=DPM_2)
    dpm2_a: DPM_2_a = field(default_factory=DPM_2_a)
    dpmpp_2s_a: DPMpp_2S_a = field(default_factory=DPMpp_2S_a)
    dpmpp_2m: DPMpp_2M = field(default_factory=DPMpp_2M)
    dpmpp_sde: DPMpp_SDE = field(default_factory=DPMpp_SDE)
    dpmpp_2m_sde: DPMpp_2M_SDE = field(default_factory=DPMpp_2M_SDE)
    dpmpp_3m_sde: DPMpp_3M_SDE = field(default_factory=DPMpp_3M_SDE)
