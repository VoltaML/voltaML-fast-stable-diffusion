import logging
from os import makedirs
from pathlib import Path
from typing import List, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.types import Img2ImgQueueEntry, InpaintQueueEntry, Txt2ImgQueueEntry

logger = logging.getLogger(__name__)


def create_metadata(
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
    ]
):
    "Return image with metadata burned into it"

    data = job.data
    metadata = PngInfo()

    def write_metadata(key: str):
        metadata.add_text(key, data.__dict__.get(key, ""))

    for i in [
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "seed",
        "strength",
    ]:
        write_metadata(i)

    procedure = ""
    if isinstance(job, Txt2ImgQueueEntry):
        procedure = "txt2img"
    elif isinstance(job, Img2ImgQueueEntry):
        procedure = "img2img"
    elif isinstance(job, InpaintQueueEntry):
        procedure = "inpaint"

    metadata.add_text("procedure", procedure)
    metadata.add_text("model", job.model)

    return metadata


def save_images(
    images: List[Image.Image],
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
    ],
):
    "Save image to disk"

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

    for i, image in enumerate(images):
        path = Path(
            f"outputs/{'txt2img' if isinstance(job, Txt2ImgQueueEntry) else 'img2img'}/{prompt}/{job.data.id}-{i}.png"
        )
        makedirs(path.parent, exist_ok=True)

        metadata = create_metadata(job)

        logger.debug(f"Saving image to {path.as_posix()}")

        with path.open("wb") as f:
            image.save(f, pnginfo=metadata)
