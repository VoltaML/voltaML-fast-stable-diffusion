import logging
import os
from pathlib import Path
from typing import Dict

from PIL import Image

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
