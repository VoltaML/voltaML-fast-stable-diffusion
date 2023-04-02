from core.config import config
from core.files import CachedModelList
from core.gpu import GPU

cached_model_list = CachedModelList()
gpu = GPU(config.api.device_id)
