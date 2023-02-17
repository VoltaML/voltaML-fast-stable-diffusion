from typing import List

from fastapi import APIRouter, HTTPException
from PIL import Image

from core.errors import ModelNotLoadedError
from core.shared_dependent import cluster
from core.types import (
    BuildRequest,
    ImageVariationsQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Txt2ImgQueueEntry,
)
from core.utils import convert_bytes_to_image_stream, convert_image_to_base64

router = APIRouter(tags=["txt2img"])


@router.post("/txt2img")
async def txt2img_job(job: Txt2ImgQueueEntry):
    "Generate images from text"

    try:
        images: List[Image.Image]
        time: float
        images, time = await cluster.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/img2img")
async def img2img_job(job: Img2ImgQueueEntry):
    "Modify image with prompt"

    data = job.data.image
    assert isinstance(data, bytes)
    job.data.image = convert_bytes_to_image_stream(data)

    try:
        images: List[Image.Image]
        time: float
        images, time = await cluster.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/inpainting")
async def inpaint_job(job: InpaintQueueEntry):
    "Inpaint image with prompt"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    mask_bytes = job.data.mask_image
    assert isinstance(mask_bytes, bytes)
    job.data.mask_image = convert_bytes_to_image_stream(mask_bytes)

    try:
        images: List[Image.Image]
        time: float
        images, time = await cluster.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/image_variations")
async def image_variations_job(job: ImageVariationsQueueEntry):
    "Generate variations of the image"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    try:
        images: List[Image.Image]
        time: float
        images, time = await cluster.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/generate-engine")
async def generate_engine(request: BuildRequest):
    "Generate a TensorRT engine from a local model"

    await cluster.build_engine(request)

    return {"message": "Success"}


@router.post("/to-fp16")
async def to_fp16(model: str):
    "Cast a model to Float16 and save it"

    await cluster.to_fp16(model)

    return {"message": "Success"}
