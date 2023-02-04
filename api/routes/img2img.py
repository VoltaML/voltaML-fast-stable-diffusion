import re
from typing import List

from fastapi import APIRouter, HTTPException
from PIL import Image

from core.errors import BadSchedulerError, ModelNotLoadedError
from core.shared_dependent import cluster
from core.types import Img2ImgQueueEntry
from core.utils import convert_image_to_base64

router = APIRouter(tags=["img2img"])


@router.post("/generate")
async def img2img_job(job: Img2ImgQueueEntry):
    "Generate images from text"

    pattern = re.compile(r"data:image\/[\w]+;base64,")

    img = job.data.image
    if isinstance(img, bytes):
        img = img.decode("utf-8")

    if isinstance(img, str):
        img = re.sub(pattern, "", img)
    job.data.image = img

    try:
        images: List[Image.Image]
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
