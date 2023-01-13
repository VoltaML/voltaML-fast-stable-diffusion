from typing import TYPE_CHECKING, Union

from core.inference.pytorch import PyTorchInferenceModel

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion


current_model: Union["DemoDiffusion", PyTorchInferenceModel, None] = None
current_steps: int = 50
