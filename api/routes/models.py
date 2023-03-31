import logging
import traceback

import torch
from fastapi import APIRouter, HTTPException

from core.shared_dependent import cached_model_list, gpu
from core.types import InferenceBackend

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/loaded")
async def list_loaded_models():
    "Returns a list containing information about loaded models"

    return gpu.loaded_models


@router.get("/avaliable")
async def list_avaliable_models():
    "Show a list of avaliable models"

    return cached_model_list.all()


@router.post("/load")
async def load_model(
    model: str,
    backend: InferenceBackend,
):
    "Loads a model into memory"

    try:
        await gpu.load_model(model, backend)
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
    return {"message": "Model unloaded"}


@router.post("/unload-all")
async def unload_all_models():
    "Unload all models from memory"

    await gpu.unload_all()

    return {"message": "All models unloaded"}


@router.post("/memory-cleanup")
async def cleanup():
    "Free up memory manually"

    gpu.memory_cleanup()
    return {"message": "Memory cleaned up"}


@router.post("/download")
async def download_model(model: str):
    "Download a model to the cache"

    await gpu.download_huggingface_model(model)
    return {"message": "Model downloaded"}
