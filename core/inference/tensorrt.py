from core.types import Job

from .base_model import InferenceModel


class TensorRTModel(InferenceModel):
    "High level wrapper for the TensorRT model"

    def __init__(self, model_id: str, use_f32: bool = False, device: str = "cuda"):
        self.model_id = model_id
        self.use_f32 = use_f32
        self.device = device

    def load(self):
        "Loads the model into the memory"

    def unload(self):
        "Unloads the model from the memory"

    def generate(self, job: Job):
        "Generates the output for the given job"
