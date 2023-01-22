import logging
from os import makedirs
from pathlib import Path
from typing import Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.types import Img2ImgQueueEntry, Txt2ImgQueueEntry

logger = logging.getLogger(__name__)


def create_metadata(job: Union[Txt2ImgQueueEntry, Img2ImgQueueEntry]):
    "Return image with metadata burned into it"

    metadata = PngInfo()
    metadata.add_text("prompt", job.data.prompt)
    metadata.add_text("negative_prompt", job.data.negative_prompt)
    metadata.add_text("width", str(job.data.width))
    metadata.add_text("height", str(job.data.height))
    metadata.add_text("steps", str(job.data.steps))
    metadata.add_text("guidance_scale", str(job.data.guidance_scale))
    metadata.add_text("seed", str(job.data.seed))
    metadata.add_text("model", job.model)

    return metadata


def save_image(image: Image.Image, job: Union[Txt2ImgQueueEntry, Img2ImgQueueEntry]):
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

    path = Path(
        f"outputs/{'txt2img' if isinstance(job, Txt2ImgQueueEntry) else 'img2img'}/{prompt}/{job.data.id}.png"
    )
    makedirs(path.parent, exist_ok=True)

    metadata = create_metadata(job)

    logger.debug(f"Saving image to {path.as_posix()}")

    with path.open("wb") as f:
        image.save(f, pnginfo=metadata)
