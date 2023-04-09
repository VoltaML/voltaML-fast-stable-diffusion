from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, Literal, List, Union
import shutil

import requests
from PIL import Image

from core.files import get_full_model_path

logger = logging.getLogger(__name__)


def image_meta_from_file(path: Path) -> Dict[str, str]:
    "Return image metadata from a file"

    with path.open("rb") as f:
        image = Image.open(f)
        text = image.text  # type: ignore

        return text


@dataclass
class CivitAiModel:
    name: str
    creator: str
    id: str # repository
    link: str
    tags: List[str]
    download_link: str


def fetch_civitai_models(t: Union[List[str], Literal["Checkpoint", "LORA", "Controlnet"]] = "Checkpoint", page: int = 1) -> List[CivitAiModel]:
    params = {}
    if isinstance(t, list):
        params["types[]"] = ",".join(t)
    else:
        params["types"] = t
    params["page"] = page
    params["limit"] = 9
    resp = requests.get("https://civitai.com/api/v1/models", params=params).json()["items"] # pylint: disable=missing-timeout
    f = []
    for item in resp:
        name = item["name"]
        id = item["id"]
        tags: List[str] = item["tags"]
        creator = item["creator"]["username"]
        tags.append(item["type"].lower())
        try:
            link = f"https://civitai.com/models/{item['modelId']}"
        except KeyError:
            link = f"https://civitai.com/models/{id}"
        download_link = item["modelVersions"][0]["downloadUrl"]
        f.append(CivitAiModel(name, creator, id, link, tags, download_link))
    return f


def download_bits(link: Union[CivitAiModel, str]) -> Path:
    if isinstance(link, CivitAiModel):
        match link.tags[-1]:
            case "checkpoint":
                folder = "models"
            case "lora":
                folder = "lora"
            case "controlnet":
                folder = "controlnet"
            case _:
                folder = "models"
        link = link.download_link
    else:
        folder = "models"
    headers = []
    with requests.get(link, stream=True) as r: # pylint: disable=missing-timeout
        headers = r.headers
        r.close()
    file_name = headers["Content-Disposition"].split('"')[1]

    file = get_full_model_path(file_name, model_folder=folder, force=True)
    nh = {}
    if file.exists():
        nh["Range"] = f"{file.stat().st_size}-"
    else:
        open(file, "x").close() # pylint: disable=unspecified-encoding
    with requests.get(link, headers=headers, stream=True) as r: # pylint: disable=missing-timeout
        with open(file, mode="wb") as f:
            shutil.copyfileobj(r.raw, f)
    return file


def init_ait_module(
    model_name,
    workdir,
):
    from aitemplate.compiler import Model

    mod = Model(os.path.join(workdir, model_name, "test.so"))
    return mod
