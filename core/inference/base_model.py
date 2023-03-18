import gc
from abc import ABC, abstractmethod
from typing import List

import torch
from PIL import Image

from core.types import Job


class InferenceModel(ABC):
    "Base class for all models that will be used in the API"

    def __init__(self, model_id: str, use_f32: bool = False, device: str = "cuda"):
        self.model_id = model_id
        self.use_f32 = use_f32
        self.device = device

    @abstractmethod
    def load(self):
        "Loads the model into the memory"

    @abstractmethod
    def unload(self):
        "Unloads the model from the memory"

    @abstractmethod
    def generate(self, job: Job) -> List[Image.Image]:
        "Generates the output of the model"

    def cleanup(self) -> None:
        "Cleanup the GPU memory"

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
