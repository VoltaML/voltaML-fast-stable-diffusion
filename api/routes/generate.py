import logging
from typing import List, Union

from fastapi import APIRouter, HTTPException
from PIL import Image

from core.config import config
from core.errors import ModelNotLoadedError
from core.functions import images_to_response, img_to_bytes
from core.shared_dependent import gpu
from core.types import (
    AITemplateBuildRequest,
    AITemplateDynamicBuildRequest,
    ControlNetQueueEntry,
    ConvertModelRequest,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    InterrogatorQueueEntry,
    ONNXBuildRequest,
    Txt2ImgQueueEntry,
    UpscaleQueueEntry,
)
from core.utils import convert_bytes_to_image_stream, convert_image_to_base64

router = APIRouter(tags=["generate"])
logger = logging.getLogger(__name__)


@router.post("/txt2img")
def txt2img_job(job: Txt2ImgQueueEntry):
    "Generate images from text"

    try:
        images: Union[List[Image.Image], List[str]]
        time: float
        images, time = gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    return images_to_response(images, time)


@router.post("/img2img")
def img2img_job(job: Img2ImgQueueEntry):
    "Modify image with prompt"

    data = job.data.image
    assert isinstance(data, bytes)
    job.data.image = convert_bytes_to_image_stream(data)

    try:
        images: Union[List[Image.Image], List[str]]
        time: float
        images, time = gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    return images_to_response(images, time)


@router.post("/inpainting")
def inpaint_job(job: InpaintQueueEntry):
    "Inpaint image with prompt"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    mask_bytes = job.data.mask_image
    assert isinstance(mask_bytes, bytes)
    job.data.mask_image = convert_bytes_to_image_stream(mask_bytes)

    try:
        images: Union[List[Image.Image], List[str]]
        time: float
        images, time = gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    return images_to_response(images, time)


@router.post("/controlnet")
def controlnet_job(job: ControlNetQueueEntry):
    "Generate images based on a reference image"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    try:
        images: Union[List[Image.Image], List[str]]
        time: float
        images, time = gpu.generate(job)
    except ModelNotLoadedError:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    return images_to_response(images, time)


@router.post("/upscale")
def realesrgan_upscale_job(job: UpscaleQueueEntry):
    "Upscale image with RealESRGAN model"

    image_bytes = job.data.image
    assert isinstance(image_bytes, bytes)
    job.data.image = convert_bytes_to_image_stream(image_bytes)

    try:
        image: Image.Image
        time: float
        image, time = gpu.upscale(job)
    except ModelNotLoadedError:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    return {
        "time": time,
        "images": convert_image_to_base64(image)
        if config.api.image_return_format == "base64"
        else img_to_bytes(image),
    }


@router.post("/generate-aitemplate")
def generate_aitemplate(request: AITemplateBuildRequest):
    "Generate an AITemplate model from a local model"

    gpu.build_aitemplate_engine(request)

    return {"message": "Success"}


@router.post("/generate-dynamic-aitemplate")
def generate_dynamic_aitemplate(request: AITemplateDynamicBuildRequest):
    "Generate an AITemplate engine from a local model"

    gpu.build_dynamic_aitemplate_engine(request)

    return {"message": "Success"}


@router.post("/generate-onnx")
def generate_onnx(request: ONNXBuildRequest):
    "Generate an ONNX model from a local model"

    gpu.build_onnx_engine(request)

    return {"message": "Success"}


@router.post("/convert-model")
def convert_model(request: ConvertModelRequest):
    "Convert a Stable Diffusion model"

    gpu.convert_model(model=request.model, safetensors=request.safetensors)

    return {"message": "Success"}


@router.post("/interrogate")
def interrogate(request: InterrogatorQueueEntry):
    "Interrogate a model"

    data = request.data.image
    assert isinstance(data, bytes)
    request.data.image = convert_bytes_to_image_stream(data)

    result = gpu.interrogate(request)
    return result
