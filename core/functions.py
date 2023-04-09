from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, Literal, List, Union
from tqdm import tqdm

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


def fetch_civitai_models(t: Union[List[str], Literal["Checkpoint", "LORA"]] = "Checkpoint", page: int = 1) -> List[CivitAiModel]:
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


def download_model(link: Union[CivitAiModel, str]) -> Path:
    folder = "models"
    if isinstance(link, CivitAiModel):
        match link.tags[-1]:
            case "checkpoint":
                folder = "models"
            case "lora":
                folder = "lora"
        link = link.download_link

    with requests.get(link, stream=True) as r:  # pylint: disable=missing-timeout
        file_name = r.headers["Content-Disposition"].split('"')[1]
        file = get_full_model_path(file_name, model_folder=folder, force=True)
        cl = int(r.headers["Content-Length"])
        logger.info("Downloading %s", file_name)
        with tqdm(total=cl, unit="iB", unit_scale=True) as pb:
            # AFAIK Windows doesn't like big buffers
            s = (64 if os.name == "nt" else 1024) * 1024
            with open(file, mode="wb+") as f:
                for data in r.iter_content(s):
                    pb.update(len(data))
                    f.write(data)
    return file


def init_ait_module(
    model_name,
    workdir,
):
    from aitemplate.compiler import Model

    mod = Model(os.path.join(workdir, model_name, "test.so"))
    return mod
