from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from core.files import CachedModelList
from core.gpu import GPU

if TYPE_CHECKING:
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        LineartAnimeDetector,
        LineartDetector,
        MidasDetector,
        MLSDdetector,
        NormalBaeDetector,
        OpenposeDetector,
    )
    from transformers.models.upernet import UperNetForSemanticSegmentation

    from core.extra.cloudflare_r2 import R2Bucket


disable_hardware_warning: bool = False
cached_model_list = CachedModelList()
gpu = GPU()
cached_controlnet_preprocessor: Union[
    None,
    "CannyDetector",
    "MidasDetector",
    "HEDdetector",
    "MLSDdetector",
    "OpenposeDetector",
    "NormalBaeDetector",
    "LineartDetector",
    "LineartAnimeDetector",
    Tuple["Any", "UperNetForSemanticSegmentation"],
] = None

r2: Optional["R2Bucket"] = None
