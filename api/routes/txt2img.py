import logging
from typing import List

from fastapi import APIRouter, HTTPException
from PIL.Image import Image

from api.shared import state
from core import cluster
from core.errors import BadSchedulerError, ModelNotLoadedError
from core.types import Txt2ImgQueueEntry
from core.utils import convert_image_to_base64

router = APIRouter()


@router.post("/interupt")
async def stop():
    "Interupt the current job"

    state.interrupt = True
    return {"message": "Interupted"}


@router.post("/generate")
async def txt2img_job(job: Txt2ImgQueueEntry):
    "Generate images from text"

    logging.debug(f"Job: {job}")

    try:
        images: List[Image]
        time: float
        images, time = await cluster.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )
    except BadSchedulerError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Scheduler is not of a proper type"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }
