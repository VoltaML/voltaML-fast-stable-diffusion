import faulthandler
import logging

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/alive")
def test():
    return {"message": "Hello World"}


@router.get("/huggingface-models.json")
def huggingface_models():
    return FileResponse("static/huggingface-models.json")


@router.post("/dump-thread-traceback")
def dump_thread_traceback():
    faulthandler.dump_traceback(all_threads=True)
