import logging
from typing import Literal, Optional

import torch
from fastapi import APIRouter, HTTPException

from core import cached_model_list, cluster
from core.gpu import GPU

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/loaded")
async def list_loaded_models():
    "Returns a list containing information about loaded models"

    models = await cluster.loaded_models()
    logger.debug(models)
    return models


@router.get("/avaliable")
async def list_avaliable_models():
    "Show a list of avaliable models"

    return [i for i in cached_model_list.pytorch()]


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
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=500, detail="Out of memory"
        )
    return {"message": "Model loaded"}


@router.post("/unload")
async def unload_model(model: str, gpu_id: int):
    "Unloads a model from memory"

    gpu: GPU = [i for i in cluster.gpus if i.gpu_id == gpu_id][0]
    await gpu.unload(model)
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
