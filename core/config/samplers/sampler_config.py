from dataclasses import dataclass, field
from typing import List, Literal, Union

from .kdiffusion_sampler_config import DPM_2S_a


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
    sigma_always_discard_next_to_last: FrontendComponent = field(
        default_factory=Checkbox
    )
    sigma_rho: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sigma_min: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sigma_max: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sampler_eta: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sampler_churn: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sampler_tmin: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sampler_tmax: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )
    sampler_noise: FrontendComponent = field(
        default_factory=lambda: NumberInput(min=0, max=1, step=0.01)
    )


@dataclass
class SamplerConfig:
    # Settings for UI
    ui_settings: ParamSettings = field(default_factory=ParamSettings)

    # Samplers
    dpmpp_2s_a: DPM_2S_a = field(default_factory=DPM_2S_a)
