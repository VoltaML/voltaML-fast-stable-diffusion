from dataclasses import dataclass, field

from core.flags import HighResFixFlag, XLFlag, XLRefinerFlag


@dataclass
class FlagsConfig:
    "Configuration for flags"

    highres: HighResFixFlag = field(default_factory=HighResFixFlag)
    refiner: XLRefinerFlag = field(default_factory=XLRefinerFlag)
    xl: XLFlag = field(default_factory=XLFlag)
