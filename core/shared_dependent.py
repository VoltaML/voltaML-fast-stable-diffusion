from typing import TYPE_CHECKING, Optional

from core.config import config
from core.files import CachedModelList
from core.gpu import GPU

if TYPE_CHECKING:
    from core.extra.cloudflare_r2 import R2Bucket

cached_model_list = CachedModelList()
gpu = GPU(config.api.device_id)

r2: Optional["R2Bucket"] = None
