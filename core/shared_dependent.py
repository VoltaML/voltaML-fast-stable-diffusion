from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from core.config import config
from core.files import CachedModelList
from core.gpu import GPU

if TYPE_CHECKING:
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        MidasDetector,
        MLSDdetector,
        OpenposeDetector,
    )
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    from core.extra.cloudflare_r2 import R2Bucket


cached_model_list = CachedModelList()
gpu = GPU(config.api.device_id)
cached_controlnet_preprocessor: Union[
    None,
    "CannyDetector",
    "MidasDetector",
    "HEDdetector",
    "MLSDdetector",
    "OpenposeDetector",
    Tuple["Any", "UperNetForSemanticSegmentation"],
] = None

r2: Optional["R2Bucket"] = None
