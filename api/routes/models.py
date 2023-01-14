import logging
from typing import Literal

import torch
from fastapi import APIRouter, HTTPException

from core import queue
from core.inference.pytorch import PyTorchInferenceModel
from core.types import SupportedModel

router = APIRouter(tags=["models"])


@router.get("/loaded")
async def list_loaded_models():
    "Returns a dictionary containing information about loaded models"

    models = queue.model_handler.generated_models
    loaded_models = {}

    for model in models:
        logging.debug(f"Model: {model}")
        logging.debug(f"Backend: {models[model]}")
        hl_model = models[model]

        if isinstance(hl_model, PyTorchInferenceModel):
            ll_model = hl_model.model
            assert ll_model is not None
        else:
            ll_model = hl_model

        scheduler = ll_model.scheduler  # type: ignore
        logging.debug(f"Current scheduler: {type(scheduler).__name__}")
        loaded_models[model] = {
            "backend": "PyTorch"
            if isinstance(models[model], PyTorchInferenceModel)
            else "TensorRT",
            "current_scheduler": type(scheduler).__name__,
            "device": models[model].device,
        }

    return loaded_models


@router.get("/avaliable")
async def list_avaliable_models():
    "Show a list of avaliable models"

    return [i.value for i in SupportedModel]


@router.post("/load")
async def load_model(
    model: str, backend: Literal["PyTorch", "TensorRT"], device: str = "cuda"
):
    "Loads a model into memory"

    try:
        await queue.load_model(model, backend, device)
    except torch.cuda.OutOfMemoryError:  # type: ignore
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=500, detail="Out of memory"
        )
    return {"message": "Model loaded"}


@router.post("/unload")
async def unload_model(model: str):
    "Unloads a model from memory"

    queue.model_handler.unload(model)
    return {"message": "Model unloaded"}


@router.post("/cleanup")
async def cleanup():
    "Free up memory manually"

    queue.model_handler.free_memory()
    return {"message": "Memory cleaned up"}
