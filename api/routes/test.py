import logging

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/alive")
async def test():
    return {"message": "Hello World"}


@router.get("/huggingface-models.json")
async def huggingface_models():
    return FileResponse("static/huggingface-models.json")
