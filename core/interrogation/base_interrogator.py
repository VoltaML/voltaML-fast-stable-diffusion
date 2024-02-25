import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from core.config import config
from core.types import Backend, Job


@dataclass
class InterrogationResult:
    "Contains results from the interrogation"

    positive: List[Tuple[str, float]]
    negative: List[Tuple[str, float]]


class InterrogationModel(ABC):
    "Base class for all interrogator models that will be used in the API"

    def __init__(self, device: Union[str, torch.device] = "cuda"):
        self.device = device
        self.backend: Backend = "unknown"

    @abstractmethod
    def load(self):
        "Loads the model into the memory"

    @abstractmethod
    def unload(self):
        "Unloads the model from the memory"

    @abstractmethod
    def generate(self, job: Job) -> InterrogationResult:
        "Generates the output of the model"

    def memory_cleanup(self) -> None:
        "Cleanup the GPU memory"

        if "cpu" in config.api.device or "privateuseone" in config.api.device:
            gc.collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
