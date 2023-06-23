import logging
import os
import shutil
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import torch
from fastapi import APIRouter, HTTPException, Request, UploadFile
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget

from api import websocket_manager
from api.websockets.data import Data
from core.files import get_full_model_path
from core.shared_dependent import cached_model_list, gpu
from core.types import (
    DeleteModelRequest,
    InferenceBackend,
    LoraLoadRequest,
    ModelResponse,
    TextualInversionLoadRequest,
)

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)

model_upload_dir = Path("data/models")
lora_upload_dir = Path("data/lora")
textual_inversions_UploadDir = Path("data/textual-inversion")


class UploadFileTarget(FileTarget):
    "A target that writes to a temporary file and then moves it to the target dir"

    def __init__(self, dir_: Path, *args, **kwargs):
        super().__init__(None, *args, **kwargs)  # type: ignore
        self.file = UploadFile(
            filename=None, file=NamedTemporaryFile(delete=False, dir=dir_)  # type: ignore
        )
        self._fd = self.file.file

    def on_start(self):
        self.file.filename = self.filename = self.multipart_filename  # type: ignore
        if model_upload_dir.joinpath(self.filename).exists():  # type: ignore
            raise HTTPException(409, "File already exists")


@router.get("/loaded")
async def list_loaded_models() -> List[ModelResponse]:
    "Returns a list containing information about loaded models"

    loaded_models = []
    for model_id in gpu.loaded_models:
        loaded_models.append(
            ModelResponse(
                name=model_id,
                backend=gpu.loaded_models[model_id].backend,
                path=gpu.loaded_models[model_id].model_id,
                state="loaded",
                loras=gpu.loaded_models[model_id].__dict__.get("loras", []),
                textual_inversions=gpu.loaded_models[model_id].__dict__.get(
                    "textual_inversions", []
                ),
                valid=True,
            )
        )

    return loaded_models


@router.get("/available")
async def list_available_models() -> List[ModelResponse]:
    "Show a list of available models"

    return cached_model_list.all()


@router.post("/load")
async def load_model(
    model: str,
    backend: InferenceBackend,
):
    "Loads a model into memory"

    try:
        await gpu.load_model(model, backend)

        await websocket_manager.broadcast(
            data=Data(data_type="refresh_models", data={})
        )
    except torch.cuda.OutOfMemoryError:  # type: ignore
        logger.warning(traceback.format_exc())
        raise HTTPException(  # pylint: disable=raise-missing-from
            status_code=500, detail="Out of memory"
        )
    return {"message": "Model loaded"}


@router.post("/unload")
async def unload_model(model: str):
    "Unloads a model from memory"

    await gpu.unload(model)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model unloaded"}


@router.post("/unload-all")
async def unload_all_models():
    "Unload all models from memory"

    await gpu.unload_all()
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))

    return {"message": "All models unloaded"}


@router.post("/load-lora")
async def load_lora(req: LoraLoadRequest):
    "Load a LoRA model into a model"

    await gpu.load_lora(req)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "LoRA model loaded"}


@router.post("/load-textual-inversion")
async def load_textual_inversion(req: TextualInversionLoadRequest):
    "Load a LoRA model into a model"

    await gpu.load_textual_inversion(req)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "LoRA model loaded"}


@router.post("/memory-cleanup")
async def cleanup():
    "Free up memory manually"

    gpu.memory_cleanup()
    return {"message": "Memory cleaned up"}


@router.post("/download")
async def download_model(model: str):
    "Download a model to the cache"

    await gpu.download_huggingface_model(model)
    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model downloaded"}


@router.get("/current-cached-preprocessor")
async def get_current_cached_preprocessor():
    "Get the current cached preprocessor"

    from core import shared_dependent

    if not shared_dependent.cached_controlnet_preprocessor:
        return {
            "preprocessor": shared_dependent.cached_controlnet_preprocessor.__class__.__name__
            if shared_dependent.cached_controlnet_preprocessor
            else None
        }
    else:
        return {"preprocessor": None}


@router.post("/upload-model")
async def upload_model(request: Request):
    "Upload a model file to the server"

    upload_type = request.query_params.get("type", "model")
    logger.info(f"Recieving model of type {upload_type}")

    parser = StreamingFormDataParser(request.headers)
    target = UploadFileTarget(model_upload_dir)
    try:
        parser.register("file", target)

        async for chunk in request.stream():
            parser.data_received(chunk)

        if target.filename:
            if upload_type == "lora":
                logger.info("Moving file to lora upload dir")
                folder = lora_upload_dir
            elif upload_type == "textual-inversion":
                logger.info("Moving file to textual inversion upload dir")
                folder = textual_inversions_UploadDir
            elif upload_type == "model":
                logger.info("Moving file to model upload dir")
                folder = model_upload_dir
            else:
                raise HTTPException(422, "Invalid upload type")

            shutil.move(target.file.file.name, folder.joinpath(target.filename))
        else:
            raise HTTPException(422, "Could not find file in body")
    finally:
        await target.file.close()
        if os.path.exists(target.file.file.name):
            os.unlink(target.file.file.name)

        await websocket_manager.broadcast(
            data=Data(data_type="refresh_models", data={})
        )
    return {"message": "Model uploaded"}


@router.delete("/delete-model")
async def delete_model(req: DeleteModelRequest):
    "Delete a model from the server"

    if req.model_type == "pytorch":
        path = get_full_model_path(req.model_path, diffusers_skip_ref_follow=True)
    elif req.model_type == "lora":
        path = lora_upload_dir.joinpath(req.model_path)
    elif req.model_type == "textual-inversion":
        path = textual_inversions_UploadDir.joinpath(req.model_path)
    else:
        raise HTTPException(422, "Invalid model type")

    logger.warning(f"Deleting model {path} of type {req.model_type}")

    if not path.exists():
        raise HTTPException(404, "Model not found")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        os.unlink(path)

    await websocket_manager.broadcast(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model deleted"}
