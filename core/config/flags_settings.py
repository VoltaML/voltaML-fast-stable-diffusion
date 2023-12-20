from dataclasses import dataclass, field

from core.flags import ADetailerFlag, SDXLFlag, SDXLRefinerFlag


@dataclass
class FlagsConfig:
    "Configuration for flags"

    refiner: SDXLRefinerFlag = field(default_factory=SDXLRefinerFlag)
    sdxl: SDXLFlag = field(default_factory=SDXLFlag)
    adetailer: ADetailerFlag = field(default_factory=ADetailerFlag)
