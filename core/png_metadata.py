import copy
import logging
from dataclasses import fields
from io import BytesIO
from os import makedirs
from pathlib import Path
from typing import List, Union

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.config import config
from core.types import (
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    SDUpscaleQueueEntry,
    Txt2ImgQueueEntry,
    UpscaleQueueEntry,
)
from core.utils import unwrap_enum_name

logger = logging.getLogger(__name__)


def create_metadata(
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
        ControlNetQueueEntry,
        SDUpscaleQueueEntry,
        UpscaleQueueEntry,
    ],
    index: int,
):
    "Return image with metadata burned into it"

    data = copy.copy(job.data)
    metadata = PngInfo()

    if not isinstance(job, UpscaleQueueEntry):
        data.seed = str(job.data.seed) + (f"({index})" if index > 0 else "")  # type: ignore Overwrite for sequencialy generated images

    def write_metadata(key: str):
        metadata.add_text(key, str(unwrap_enum_name(data.__dict__.get(key, ""))))

    for key in fields(data):
        if key.name not in ("image", "mask_image"):
            write_metadata(key.name)

    if isinstance(job, Txt2ImgQueueEntry):
        procedure = "txt2img"
    elif isinstance(job, Img2ImgQueueEntry):
        procedure = "img2img"
    elif isinstance(job, InpaintQueueEntry):
        procedure = "inpaint"
    elif isinstance(job, ControlNetQueueEntry):
        procedure = "control_net"
    elif isinstance(job, UpscaleQueueEntry):
        procedure = "upscale"
    else:
        procedure = "unknown"

    metadata.add_text("procedure", procedure)
    metadata.add_text("model", job.model)

    return metadata


def save_images(
    images: List[Image.Image],
    job: Union[
        Txt2ImgQueueEntry,
        Img2ImgQueueEntry,
        InpaintQueueEntry,
        ControlNetQueueEntry,
        UpscaleQueueEntry,
        SDUpscaleQueueEntry,
    ],
):
    "Save image to disk or r2"

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

    urls: List[str] = []
    for i, image in enumerate(images):
        if isinstance(job, (UpscaleQueueEntry, SDUpscaleQueueEntry)):
            folder = "extra"
        elif isinstance(job, Txt2ImgQueueEntry):
            folder = "txt2img"
        else:
            folder = "img2img"

        filename = f"{job.data.id}-{i}.png"
        extension = "png"
        metadata = create_metadata(job, i)

        if job.save_image == "r2":
            # Save into Cloudflare R2 bucket
            from core.shared_dependent import r2

            assert r2 is not None, "R2 is not configured, enable debug mode to see why"

            image_bytes = BytesIO()
            image.save(image_bytes, pnginfo=metadata, format="png")
            image_bytes.seek(0)

            url = r2.upload_file(file=image_bytes, filename=filename)
            if url:
                logger.debug(f"Saved image to R2: {filename}")
                urls.append(url)
            else:
                logger.debug("No provided Dev R2 URL, uploaded but returning empty URL")
        else:
            base_dir = Path("data/outputs")
            extra_path = config.api.save_path_template.format(
                **{
                    "prompt": prompt,
                    "id": job.data.id,
                    "folder": folder,
                    "seed": job.data.seed
                    if not isinstance(job, UpscaleQueueEntry)
                    else "0",
                    "index": i,
                    "extension": extension,
                }
            )

            path = base_dir / extra_path

            makedirs(path.parent, exist_ok=True)

            with path.open("wb") as f:
                logger.debug(f"Saving image to {path.as_posix()}")
                image.save(f, pnginfo=metadata)

    return urls
