from dataclasses import dataclass, field

from .kdiffusion_sampler_config import DPM_2S_a


@dataclass
class SamplerConfig:
    dpmpp_2s_a: DPM_2S_a = field(default_factory=DPM_2S_a)
