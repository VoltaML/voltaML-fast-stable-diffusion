import gc
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image
from fastapi import Request
from fastapi_utils.timing import record_timing

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
    def generate(self, job: Job, request: Request) -> List[Image.Image]:
        "Generates the output of the model"

    def memory_cleanup(self, request: Optional[Request]) -> None:
        "Cleanup the GPU memory"

        if config.api.device_type == "cpu" or config.api.device_type == "directml":
            gc.collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        if request is not None:
            record_timing(request, "memory clear")
