import logging
import os
from pathlib import Path
from typing import Literal

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_model(
    link: str, model_type: Literal["Checkpoint", "TextualInversion", "LORA"]
) -> Path:
    "Download a model from a link and return the path to the downloaded file."

    mtype = model_type.lower()

    if mtype == "checkpoint":
        folder = "models"
    elif mtype == "textualinversion":
        folder = "textual-inversion"
    elif mtype == "lora":
        folder = "lora"
    else:
        raise ValueError(f"Unknown model type {mtype}")

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    with session.get(link, stream=True, timeout=30) as r:
        file_name = r.headers["Content-Disposition"].split('"')[1]
        file = Path("data") / folder / file_name
        cl = int(r.headers["Content-Length"])
        logger.info(f"Downloading {file_name} into {file.as_posix()}")
        with tqdm(total=cl, unit="iB", unit_scale=True) as pb:
            # AFAIK Windows doesn't like big buffers
            s = (64 if os.name == "nt" else 1024) * 1024
            with open(file, mode="wb+") as f:
                for data in r.iter_content(s):
                    pb.update(len(data))
                    f.write(data)
    return file
