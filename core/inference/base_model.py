import gc
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from PIL import Image

from core.config import config
from core.types import Backend, Job


class InferenceModel(ABC):
    "Base class for all inference models that will be used in the API"

    def __init__(self, model_id: str, device: Union[str, torch.device] = "cuda"):
        self.model_id = model_id
        self.device = device
        self.backend: Backend = "unknown"

    @abstractmethod
    def load(self):
        "Loads the model into the memory"

    @abstractmethod
    def unload(self):
        "Unloads the model from the memory"

    @abstractmethod
    def generate(self, job: Job) -> List[Image.Image]:
        "Generates the output of the model"

    def memory_cleanup(self) -> None:
        "Cleanup the GPU memory"

        if config.api.clear_memory_policy == "always":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if torch.backends.mps.is_available():  # type: ignore
                torch.mps.empty_cache()  # type: ignore
            try:
                if torch.xpu.is_available():  # type: ignore
                    torch.xpu.empty_cache()  # type: ignore
            except AttributeError:
                pass
