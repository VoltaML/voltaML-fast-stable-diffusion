from typing import Dict

from core.inference.pytorch import PyTorchInferenceModel

from .types import SupportedModel, Txt2ImgQueueEntry


class ModelHandler:
    def __init__(self) -> None:
        self.generated_models: Dict[SupportedModel, PyTorchInferenceModel] = dict()
        
    def generate(self, job: Txt2ImgQueueEntry):
        if job.model not in self.generated_models:
            self.generated_models[job.model] = PyTorchInferenceModel(job.model.value, job.scheduler)
            self.generated_models[job.model].optimize()

        return self.generated_models[job.model].generate(job.data)
    
    def unload(self, model: SupportedModel):
        if model in self.generated_models:
            self.generated_models[model].unload()
            self.generated_models.pop(model)
    
    def unload_all(self):
        for model in self.generated_models:
            self.unload(model)