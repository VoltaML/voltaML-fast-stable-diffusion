import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import piexif
import piexif.helper
from fastapi import HTTPException
from fastapi.responses import Response
from PIL import Image
from requests import HTTPError

from core.config import config
from core.utils import convert_image_to_base64

logger = logging.getLogger(__name__)


def image_meta_from_file(path: Path) -> Dict[str, str]:
    "Return image metadata from a file"

    extension = path.suffix.lower()
    if extension == ".png":
        with path.open("rb") as f:
            image = Image.open(f)
            meta = image.text  # type: ignore

            return meta
    else:
        data = piexif.load(path.as_posix())
        meta: Dict[str, str] = json.loads(
            piexif.helper.UserComment.load(data["Exif"][piexif.ExifIFD.UserComment])
        )
        return meta


def inject_var_into_dotenv(key: str, value: str) -> None:
    """
    Injects the HuggingFace token into the .env file

    Args:
        token (str): HuggingFace token with read permissions
    """

    pattern = re.compile(f"{key}=(.*)")
    dotenv_path = Path(".env")

    if key == "HUGGINGFACE_TOKEN":
        # Check if the token is valid
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.whoami(token=value)
        except HTTPError as e:
            logger.error(f"Invalid HuggingFace token: {e}")

    if not dotenv_path.exists():
        example_dotenv = open(".env.example", "r", encoding="utf-8")
        example_dotenv_contents = example_dotenv.read()
        example_dotenv.close()

        with dotenv_path.open("w", encoding="utf-8") as f:
            f.write(example_dotenv_contents)

    with dotenv_path.open("r", encoding="utf-8") as f:
        dotenv_contents = f.read()

    dotenv_contents = pattern.sub(f"{key}={value}", dotenv_contents)

    with dotenv_path.open("w", encoding="utf-8") as f:
        f.write(dotenv_contents)

    logger.info(f"{key} was injected to the .env file")

    os.environ[key] = value
    logger.info("Variable injected into current environment")


def img_to_bytes(img: Image.Image) -> bytes:
    "Convert an image to bytes"

    with BytesIO() as output:
        img.save(output, format=config.api.image_extension)
        return output.getvalue()


def images_to_response(images: Union[List[Image.Image], List[str]], time: float):
    "Generate a valid response for the API"

    if len(images) == 0:
        return {
            "time": time,
            "images": [],
        }
    elif isinstance(images[0], str):
        return {
            "time": time,
            "images": images,
        }
    elif config.api.image_return_format == "bytes":
        if len(images) > 1:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Image return format is set to bytes, but {len(images)} images were returned"
                },
            )
        return Response(img_to_bytes(images[0]), media_type="binary/octet-stream")  # type: ignore
    else:
        return {
            "time": time,
            "images": [convert_image_to_base64(i) for i in images],  # type: ignore
        }
