from abc import ABC, abstractmethod

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
    def generate(self, job: Job) -> Job:
        "Generates the output of the model"
