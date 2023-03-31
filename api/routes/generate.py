import logging
from typing import List

from fastapi import APIRouter, HTTPException
from PIL import Image

from core.errors import ModelNotLoadedError
from core.shared_dependent import gpu
from core.types import (
    AITemplateBuildRequest,
    BuildRequest,
    ControlNetQueueEntry,
    ConvertModelRequest,
    ImageVariationsQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    RealESRGANQueueEntry,
    Txt2ImgQueueEntry,
)
from core.utils import convert_bytes_to_image_stream, convert_image_to_base64

router = APIRouter(tags=["txt2img"])
logger = logging.getLogger(__name__)


@router.post("/txt2img")
async def txt2img_job(job: Txt2ImgQueueEntry):
    "Generate images from text"

    try:
        images: List[Image.Image]
        time: float
        images, time = await gpu.generate(job)
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
        images, time = await gpu.generate(job)
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
        images, time = await gpu.generate(job)
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
        images, time = await gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/controlnet")
async def controlnet_job(job: ControlNetQueueEntry):
    "Generate variations of the image"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    try:
        images: List[Image.Image]
        time: float
        images, time = await gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=400, detail="Model is not loaded"
        )

    return {
        "time": time,
        "images": [convert_image_to_base64(i) for i in images],
    }


@router.post("/realesrgan-upscale")
async def realesrgan_upscale_job(job: RealESRGANQueueEntry):
    "Generate variations of the image"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    try:
        images: List[Image.Image]
        time: float
        images, time = await gpu.generate(job)
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

    await gpu.build_trt_engine(request)

    return {"message": "Success"}


@router.post("/generate-aitemplate")
async def generate_aitemplate(request: AITemplateBuildRequest):
    "Generate a TensorRT engine from a local model"

    await gpu.build_aitemplate_engine(request)

    return {"message": "Success"}


@router.post("/convert-model")
async def convert_model(request: ConvertModelRequest):
    "Cast a model to Float16 and save it"

    await gpu.convert_model(
        model=request.model, use_fp32=request.use_fp32, safetensors=request.safetensors
    )

    return {"message": "Success"}
