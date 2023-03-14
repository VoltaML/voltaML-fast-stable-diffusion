import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from core.functions import image_meta_from_file

router = APIRouter(tags=["output"])

thread_pool = ThreadPoolExecutor()


@router.get("/txt2img")
def txt2img() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/txt2img")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append({"path": i.as_posix(), "time": os.path.getmtime(i)})

    return data


@router.get("/img2img")
def img2img() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/img2img")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append({"path": i.as_posix(), "time": os.path.getmtime(i)})

    return data


@router.get("/extra")
def extra() -> List[Dict[str, Any]]:
    "List all generated images"

    path = Path("data/outputs/extra")

    if not path.exists():
        return []

    data: List[Dict[str, Any]] = []
    for i in path.rglob("**/*.png"):
        data.append({"path": i.as_posix(), "time": os.path.getmtime(i)})

    return data


@router.get("/data")
async def txt2img_data(filename: str) -> Dict[str, str]:
    "Get a generated image metadata"

    path = Path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return image_meta_from_file(path)
