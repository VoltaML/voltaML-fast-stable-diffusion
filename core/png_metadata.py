import copy
import logging
from os import makedirs
from pathlib import Path
from typing import List, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.types import (
    ControlNetQueueEntry,
    ImageVariationsQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Txt2ImgQueueEntry,
)

logger = logging.getLogger(__name__)


def create_metadata(
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
        ImageVariationsQueueEntry,
        ControlNetQueueEntry,
    ],
    index: int,
):
    "Return image with metadata burned into it"

    data = copy.copy(job.data)
    metadata = PngInfo()

    data.seed = str(job.data.seed) + (f"({index})" if index > 0 else "")  # type: ignore Overwrite for sequencialy generated images

    def write_metadata(key: str):
        metadata.add_text(key, str(data.__dict__.get(key, "")))

    for key in [
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "seed",
        "strength",
    ]:
        write_metadata(key)

    procedure = ""
    if isinstance(job, Txt2ImgQueueEntry):
        procedure = "txt2img"
    elif isinstance(job, Img2ImgQueueEntry):
        procedure = "img2img"
    elif isinstance(job, InpaintQueueEntry):
        procedure = "inpaint"
    elif isinstance(job, ImageVariationsQueueEntry):
        procedure = "image_variations"
    elif isinstance(job, ControlNetQueueEntry):
        procedure = "control_net"

    metadata.add_text("procedure", procedure)
    metadata.add_text("model", job.model)

    return metadata


def save_images(
    images: List[Image.Image],
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
        ImageVariationsQueueEntry,
        ControlNetQueueEntry,
    ],
):
    "Save image to disk"

    if isinstance(
        job,
        (
            Txt2ImgQueueEntry,
            Img2ImgQueueEntry,
            InpaintQueueEntry,
        ),
    ):
        prompt = (
            job.data.prompt[:30]
            .strip()
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("?", "")
            .replace("!", "")
            .replace(":", "")
            .replace(";", "")
            .replace("'", "")
            .replace('"', "")
        )
    else:
        prompt = ""

    for i, image in enumerate(images):
        path = Path(
            f"data/outputs/{'txt2img' if isinstance(job, Txt2ImgQueueEntry) else 'img2img'}/{prompt}/{job.data.id}-{i}.png"
        )
        makedirs(path.parent, exist_ok=True)

        metadata = create_metadata(job, i)

        logger.debug(f"Saving image to {path.as_posix()}")

        with path.open("wb") as f:
            image.save(f, pnginfo=metadata)
