from fastapi import APIRouter

from core import queue
from core.inference.pytorch import PyTorchInferenceModel
from core.types import SupportedModel

router = APIRouter(tags=["models"])


@router.get("/loaded")
async def list_loaded_models():
    models = queue.model_handler.generated_models
    loaded_models = {}

    for model in models:
        loaded_models[model.value] = {
            "backend": "PyTorch"
            if isinstance(models[model], PyTorchInferenceModel)
            else "TensorRT",
            "scheduler": models[model].scheduler,  # type: ignore
        }

    return loaded_models


@router.get("/avaliable")
async def list_avaliable_models():
    return [i.value for i in SupportedModel]


@router.post("/unload")
async def unload_model(model: SupportedModel):
    queue.model_handler.unload(model)
    return {"message": "Model unloaded"}
