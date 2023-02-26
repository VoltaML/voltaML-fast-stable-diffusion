import logging
import traceback
from typing import Literal, Optional

import torch
from fastapi import APIRouter, HTTPException

from core.shared_dependent import cached_model_list, cluster

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/loaded")
async def list_loaded_models():
    "Returns a list containing information about loaded models"

    models = await cluster.loaded_models()
    return models


@router.get("/avaliable")
async def list_avaliable_models():
    "Show a list of avaliable models"

    return cached_model_list.all()


@router.post("/load")
async def load_model(
    model: str,
    backend: Literal["PyTorch", "TensorRT"],
    preferred_gpu: Optional[int] = None,
):
    "Loads a model into memory"

    try:
        await cluster.load_model(model, backend, preferred_gpu=preferred_gpu)
    except torch.cuda.OutOfMemoryError:  # type: ignore
        logger.warning(traceback.format_exc())
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=500, detail="Out of memory"
        )
    return {"message": "Model loaded"}


@router.post("/unload")
async def unload_model(model: str, gpu_id: int):
    "Unloads a model from memory"

    await cluster.unload(model, gpu_id)
    return {"message": "Model unloaded"}


@router.post("/unload-all")
async def unload_all_models():
    "Unload all models from memory"

    for gpu in cluster.gpus:
        await gpu.unload_all()

    return {"message": "All models unloaded"}


@router.post("/memory-cleanup")
async def cleanup():
    "Free up memory manually"

    for gpu in cluster.gpus:
        gpu.memory_cleanup()
    return {"message": "Memory cleaned up"}


@router.post("/download")
async def download_model(model: str):
    "Download a model to the cache"

    await cluster.download_model(model)
    return {"message": "Model downloaded"}


@router.post("/convert-from-checkpoint")
async def convert_from_checkpoint(path: str, is_sd2: bool = False):
    "Convert a checkpoint to a PyTorch model"

    await cluster.convert_from_checkpoint(path, is_sd2)
    return {"message": "Model converted"}
