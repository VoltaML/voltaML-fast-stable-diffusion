import gc
from abc import ABC, abstractmethod
from typing import List

import torch
from PIL import Image

from core.config import config
from core.types import Backend, Job


class InferenceModel(ABC):
    "Base class for all inference models that will be used in the API"

    def __init__(self, model_id: str, device: str = "cuda"):
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

        if config.api.device_type == "cpu":
            gc.collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
