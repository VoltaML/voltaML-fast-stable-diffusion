import logging
import os
import re
from pathlib import Path
from typing import Dict

from PIL import Image
from requests import HTTPError

logger = logging.getLogger(__name__)


def image_meta_from_file(path: Path) -> Dict[str, str]:
    "Return image metadata from a file"

    with path.open("rb") as f:
        image = Image.open(f)
        text = image.text  # type: ignore

        return text


def init_ait_module(
    model_name,
    workdir,
):
    from aitemplate.compiler import Model

    mod = Model(os.path.join(workdir, model_name, "test.so"))
    return mod


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

    if key == "HUGGINGFACE_TOKEN":
        from core import shared

        shared.hf_token = value
