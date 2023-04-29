import logging
import traceback
from typing import List

import torch
from fastapi import APIRouter, HTTPException

from api import websocket_manager
from api.websockets.data import Data
from core.shared_dependent import cached_model_list, gpu
from core.types import InferenceBackend, ModelResponse

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/loaded")
async def list_loaded_models() -> List[ModelResponse]:
    "Returns a list containing information about loaded models"

    loaded_models = []
    for model_id in gpu.loaded_models:
        loaded_models.append(
            ModelResponse(
                name=model_id,
                backend=gpu.loaded_models[model_id].backend,
                path=gpu.loaded_models[model_id].model_id,
                state="loaded",
                loras=gpu.loaded_models[model_id].__dict__.get("loras", []),
                valid=True,
            )
        )

    return loaded_models


@router.get("/available")
async def list_available_models() -> List[ModelResponse]:
    "Show a list of available models"

    return cached_model_list.all()


@router.post("/load")
async def load_model(
    model: str,
    backend: InferenceBackend,
):
    "Loads a model into memory"

    try:
        await gpu.load_model(model, backend)

        await websocket_manager.broadcast(
            data=Data(data_type="refresh_models", data={})
        )
    except torch.cuda.OutOfMemoryError:  # type: ignore
        logger.warning(traceback.format_exc())
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=500, detail="Out of memory"
        )
    return {"message": "Model loaded"}


@router.post("/unload")
async def unload_model(model: str):
    "Unloads a model from memory"

    await gpu.unload(model)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model unloaded"}


@router.post("/unload-all")
async def unload_all_models():
    "Unload all models from memory"

    await gpu.unload_all()
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))

    return {"message": "All models unloaded"}


@router.post("/load-lora")
async def load_lora(model: str, lora: str):
    "Load a LoRA model into a model"

    await gpu.load_lora(model, lora)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "LoRA model loaded"}


@router.post("/memory-cleanup")
async def cleanup():
    "Free up memory manually"

    gpu.memory_cleanup()
    return {"message": "Memory cleaned up"}


@router.post("/download")
async def download_model(model: str):
    "Download a model to the cache"

    await gpu.download_huggingface_model(model)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model downloaded"}


@router.get("/current-cached-preprocessor")
async def get_current_cached_preprocessor():
    "Get the current cached preprocessor"

    from core import shared_dependent

    return {
        "preprocessor": shared_dependent.cached_controlnet_preprocessor.__class__.__name__
    }
