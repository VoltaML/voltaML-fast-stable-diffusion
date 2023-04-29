import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from core.functions import image_meta_from_file

router = APIRouter(tags=["output"])
thread_pool = ThreadPoolExecutor()
logger = logging.getLogger(__name__)


@router.get("/txt2img")
def txt2img() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/txt2img")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append(
            {"path": i.as_posix(), "time": os.path.getmtime(i), "id": Path(i).stem}
        )

    return data


@router.get("/img2img")
def img2img() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/img2img")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append(
            {"path": i.as_posix(), "time": os.path.getmtime(i), "id": Path(i).stem}
        )

    return data


@router.get("/extra")
def extra() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/extra")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append(
            {"path": i.as_posix(), "time": os.path.getmtime(i), "id": Path(i).stem}
        )

    return data


@router.get("/data")
async def image_data(filename: str) -> Dict[str, str]:
    "Get a generated image metadata"

    path = Path(filename)
    path_str = path.as_posix()

    # CodeQl: Path Traversal fix
    if not path_str.startswith("data/outputs"):
        raise HTTPException(status_code=403, detail="Access denied")

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return image_meta_from_file(path)


@router.delete("/delete")
async def delete_image(filename: str) -> Dict[str, str]:
    "Delete a generated image (does not purge the directory)"

    path = Path(filename)
    path_str = path.as_posix()

    # CodeQl: Path Traversal fix
    if not path_str.startswith("data/outputs"):
        raise HTTPException(status_code=403, detail="Access denied")

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    path.unlink()

    return {"message": "File deleted"}
