from dataclasses import dataclass, field

from core.flags import HighResFixFlag, SDXLFlag, SDXLRefinerFlag


@dataclass
class FlagsConfig:
    "Configuration for flags"

    highres: HighResFixFlag = field(default_factory=HighResFixFlag)
    refiner: SDXLRefinerFlag = field(default_factory=SDXLRefinerFlag)
    sdxl: SDXLFlag = field(default_factory=SDXLFlag)
