from dataclasses import dataclass, field

from core.flags import SDXLFlag, SDXLRefinerFlag


@dataclass
class FlagsConfig:
    "Configuration for flags"

    refiner: SDXLRefinerFlag = field(default_factory=SDXLRefinerFlag)
    sdxl: SDXLFlag = field(default_factory=SDXLFlag)
