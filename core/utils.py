import asyncio
import base64
import json
import logging
import math
import os
import re
import struct
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from core.thread import ThreadWithReturnValue
from core.types import (
    ControlNetQueueEntry,
    ImageFormats,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    PyTorchModelBase,
    PyTorchModelStage,
    Txt2ImgQueueEntry,
)

logger = logging.getLogger(__name__)
content_disposition_regex = re.compile(r"filename=[\"]?([^\";\n]+)[\"]?")


def unwrap_enum(possible_enum: Union[Enum, Any]) -> Any:
    "Unwrap an enum to its value"

    if isinstance(possible_enum, Enum):
        return possible_enum.value
    return possible_enum


def unwrap_enum_name(possible_enum: Union[Enum, Any]):
    "Unwrap an enum to its name"

    if isinstance(possible_enum, Enum):
        return possible_enum.name
    return possible_enum


def get_grid_dimension(length: int) -> Tuple[int, int]:
    "Generate the dimensions of a grid so that images can be tiled"

    cols = math.ceil(length**0.5)
    rows = math.ceil(length / cols)
    return cols, rows


def convert_image_to_stream(
    image: Union[BytesIO, Image.Image],
    quality: int = 95,
    _format: ImageFormats = "webp",
) -> BytesIO:
    "Convert an image to a stream of bytes"
    if isinstance(image, BytesIO):
        image.seek(0)
        return image

    assert image is Image.Image
    stream = BytesIO()
    image.save(stream, format=_format, quality=quality)
    stream.seek(0)
    return stream


def convert_to_image(
    image: Union[Image.Image, bytes, str], convert_to_rgb: bool = True
) -> Image.Image:
    "Converts the image to a PIL Image if it is a base64 string or bytes"

    if isinstance(image, str):
        b = convert_base64_to_bytes(image)
        im = Image.open(b)

        if convert_to_rgb:
            im = im.convert("RGB")

        return im

    if isinstance(image, bytes):
        im = Image.open(image)

        if convert_to_rgb:
            im = im.convert("RGB")

        return im

    if isinstance(image, Image.Image):
        return image

    raise ValueError(f"Type {type(image)} not supported yet")


def determine_model_type(
    file: Path,
) -> Tuple[str, PyTorchModelBase, PyTorchModelStage]:
    name = file.name
    model_type: PyTorchModelBase = "Unknown"
    model_stage: PyTorchModelStage = "last_stage"
    if file.suffix == ".safetensors":
        with open(file, "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            _metadata: Dict[str, Dict[str, str]] = json.loads(f.read(length))

            keys: Dict[str, str] = _metadata.get("__metadata__", {})
            if "format" in keys:
                # Model is A1111-style
                merge_recipe: str = keys.get("sd_merge_recipe", None)  # type: ignore
                if merge_recipe is not None:
                    merge_recipe_json: dict = json.loads(merge_recipe)
                    og = name
                    name = merge_recipe_json.get("custom_name", None)
                    if name is None:
                        name = og
                    else:
                        name = f"{name} ({og})"
            if (
                "conditioner.embedders.0.transformer.text_model.encoder.layers.3.layer_norm1.bias"
                in _metadata
            ):
                model_type = "SDXL"
            elif (
                "conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj_weight"
                in _metadata
            ):
                model_stage = "last_stage"
                model_type = "SDXL"
            elif (
                "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight"
                in _metadata
            ):
                model_type = "SD2.x"
            elif (
                "encoder.block.20.layer.1.DenseReluDense.wo.weight" in _metadata
                or "encoder.block.0.layer.0.SelfAttention.k.SCB" in _metadata
            ):
                model_type = "IF"
                model_stage = "text_encoding"
            elif "add_embedding.norm1.weight" in _metadata:
                model_type = "IF"
                if "class_embedding.linear_1.bias" not in _metadata:
                    model_stage = "first_stage"
    elif file.is_dir():
        if file.joinpath("model_index.json").exists():
            with open(file / "model_index.json", "r") as f:
                metadata: Dict[str, str] = json.loads(f.read())
                class_name = metadata.get("_class_name")
                if class_name == "KandinskyV22PriorPipeline":
                    model_type = "Kandinsky 2.2"
                    model_stage = "text_encoding"
                elif (
                    class_name == "KandinskyV22ControlnetPipeline"
                    or class_name == "KandinskyV22Pipeline"
                ):
                    model_type = "Kandinsky 2.2"
                elif class_name == "KandinskyPipeline":
                    model_type = "Kandinsky 2.1"
                elif class_name == "KandinskyPriorPipeline":
                    model_type = "Kandinsky 2.1"
                    model_stage = "text_encoding"
                elif class_name == "StableDiffusionPipeline":
                    # Either SD1.x or SD2.x
                    model_type = "SD1.x"
                elif class_name == "StableDiffusionXLPipeline":
                    model_type = "SDXL"
                else:
                    model_type = "Unknown"

    return (name, model_type, model_stage)


def convert_image_to_base64(
    image: Union[BytesIO, Image.Image],
    quality: int = 95,
    image_format: ImageFormats = "webp",
    prefix_js: bool = True,
) -> str:
    "Convert an image to a base64 string"

    stream = convert_image_to_stream(image, quality=quality)
    if prefix_js:
        prefix = f"data:image/{image_format};base64,"
    else:
        prefix = ""
    return prefix + base64.b64encode(stream.read()).decode("utf-8")


def convert_base64_to_bytes(data: str):
    "Convert a base64 string to bytes"

    return BytesIO(base64.b64decode(data))


async def run_in_thread_async(
    func: Union[Callable[..., Any], Coroutine[Any, Any, Any]],
    args: Optional[Tuple] = None,
    kwarkgs: Optional[Dict] = None,
) -> Any:
    "Run a function in a separate thread"

    thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwarkgs)
    thread.start()

    # wait for the thread to finish
    while thread.is_alive():
        await asyncio.sleep(0.1)

    # get the value returned from the thread
    value, exc = thread.join()

    if exc:
        raise exc

    return value


def image_grid(imgs: List[Image.Image]):
    "Make a grid of images"

    if isinstance(imgs[0], BytesIO):
        return imgs[0]

    landscape: bool = imgs[0].size[1] >= imgs[0].size[0]
    dim = get_grid_dimension(len(imgs))
    if landscape:
        cols, rows = dim
    else:
        rows, cols = dim

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def convert_images_to_base64_grid(
    images: List[Union[BytesIO, Image.Image]],
    quality: int = 95,
    image_format: ImageFormats = "png",
) -> str:
    "Convert a list of images to a list of base64 strings"

    if isinstance(images[0], BytesIO):
        quality = max(quality - 20, 20)
        return convert_image_to_base64(images[0], quality=quality, image_format="gif")
    else:
        assert images is List[Image.Image]
        return convert_image_to_base64(
            image_grid(images), quality=quality, image_format=image_format
        )


def resize(image: Image.Image, w: int, h: int):
    "Preprocess an image for the img2img procedure"

    return image.resize((w, h), resample=Image.LANCZOS)


def convert_bytes_to_image_stream(data: bytes) -> str:
    "Convert a base64 string to a PIL Image"

    pattern = re.compile(r"data:image\/[\w]+;base64,")

    img = data
    img = img.decode("utf-8")
    img = re.sub(pattern, "", img)

    return img


def download_file(url: str, file: Path, add_filename: bool = False):
    """Download a file to the specified path, or to a child of the provided file
    with the name provided in the Content-Disposition header"""

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    with session.get(url, stream=True, timeout=30) as r:
        try:
            filename_match = content_disposition_regex.search(
                r.headers["Content-Disposition"]
            )
            if filename_match:
                file_name = filename_match.group(1)
            else:
                raise KeyError
        except KeyError:
            file_name = url.split("/")[-1]

        if add_filename:
            file = file / file_name

        try:
            total = int(r.headers["Content-Length"])
        except KeyError:
            total = None
            logger.warning(
                "Content-Length header not found, progress bar will not work"
            )

        if file.exists():
            logger.debug(f"File {file.as_posix()} already exists, skipping")
            return file

        logger.info(f"Downloading {file_name} into {file.as_posix()}")
        # AFAIK Windows doesn't like big buffers
        s = (64 if os.name == "nt" else 1024) * 1024
        with open(file, mode="wb+") as f:
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for data in r.iter_content(s):
                    f.write(data)
                    pbar.update(len(data))

    return file


def preprocess_job(
    job: Union[
        Txt2ImgQueueEntry, Img2ImgQueueEntry, InpaintQueueEntry, ControlNetQueueEntry
    ]
):
    return job
