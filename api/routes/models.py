import logging
import os
import shutil
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

import torch
from fastapi import APIRouter, HTTPException, Request, UploadFile
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget

from api import websocket_manager
from api.websockets.data import Data
from api.websockets.notification import Notification
from core.files import get_full_model_path
from core.shared_dependent import cached_model_list, gpu
from core.types import (
    DeleteModelRequest,
    InferenceBackend,
    ModelResponse,
    PyTorchModelBase,
    TextualInversionLoadRequest,
    VaeLoadRequest,
)
from core.utils import determine_model_type, download_file

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)

possible_dirs = [
    "models",
    "lora",
    "textual-inversion",
    "lycoris",
    "vae",
    "aitemplate",
    "onnx",
]


class UploadFileTarget(FileTarget):
    "A target that writes to a temporary file and then moves it to the target dir"

    def __init__(self, dir_: Path, *args, **kwargs):
        super().__init__(None, *args, **kwargs)  # type: ignore
        self.file = UploadFile(
            filename=None, file=NamedTemporaryFile(delete=False, dir=dir_)  # type: ignore
        )
        self._fd = self.file.file
        self.dir = dir_

    def on_start(self):
        self.file.filename = self.filename = self.multipart_filename  # type: ignore
        if self.dir.joinpath(self.filename).exists():  # type: ignore
            raise HTTPException(409, "File already exists")


@router.get("/loaded")
def list_loaded_models() -> List[ModelResponse]:
    "Returns a list containing information about loaded models"

    loaded_models = []
    for model_id in gpu.loaded_models:
        name, type_, stage = determine_model_type(get_full_model_path(model_id))

        loaded_models.append(
            ModelResponse(
                name=name
                if (".ckpt" in model_id) or (".safetensors" in model_id)
                else model_id,
                backend=gpu.loaded_models[model_id].backend,
                path=gpu.loaded_models[model_id].model_id,
                state="loaded",
                vae=gpu.loaded_models[model_id].__dict__.get("vae_path", "default"),
                textual_inversions=gpu.loaded_models[model_id].__dict__.get(
                    "textual_inversions", []
                ),
                valid=True,
                stage=stage,
                type=type_,
            )
        )

    return loaded_models


@router.get("/available")
def list_available_models() -> List[ModelResponse]:
    "Show a list of available models"

    return cached_model_list.all()


@router.post("/load")
def load_model(
    model: str,
    backend: InferenceBackend,
    type: PyTorchModelBase,
):
    "Loads a model into memory"

    try:
        gpu.load_model(model, backend, type)

        websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    except torch.cuda.OutOfMemoryError:  # type: ignore
        logger.warning(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Out of memory")
    return {"message": "Model loaded"}


@router.post("/unload")
def unload_model(model: str):
    "Unloads a model from memory"

    gpu.unload(model)
    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model unloaded"}


@router.post("/unload-all")
def unload_all_models():
    "Unload all models from memory"

    gpu.unload_all()
    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))

    return {"message": "All models unloaded"}


@router.post("/load-vae")
def load_vae(req: VaeLoadRequest):
    "Load a VAE into a model"

    gpu.load_vae(req)
    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    return {"message": "VAE model loaded"}


@router.post("/load-textual-inversion")
def load_textual_inversion(req: TextualInversionLoadRequest):
    "Load a LoRA model into a model"

    gpu.load_textual_inversion(req)
    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    return {"message": "LoRA model loaded"}


@router.post("/memory-cleanup")
def cleanup():
    "Free up memory manually"

    gpu.memory_cleanup()
    return {"message": "Memory cleaned up"}


@router.post("/download")
def download_model(model: str):
    "Download a model to the cache"

    gpu.download_huggingface_model(model)
    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model downloaded"}


@router.get("/current-cached-preprocessor")
def get_current_cached_preprocessor():
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

    upload_type = request.query_params.get("type")
    assert upload_type in possible_dirs, f"Invalid upload type '{upload_type}'"

    logger.info(f"Recieving model of type '{upload_type}'")

    upload_dir = Path("data") / upload_type

    parser = StreamingFormDataParser(request.headers)
    target = UploadFileTarget(upload_dir)
    try:
        parser.register("file", target)

        async for chunk in request.stream():
            parser.data_received(chunk)

        if target.filename:
            shutil.move(target.file.file.name, upload_dir.joinpath(target.filename))
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
def delete_model(req: DeleteModelRequest):
    "Delete a model from the server"

    assert req.model_type in possible_dirs, f"Invalid upload type {req.model_type}"

    if req.model_type == "models":
        path = get_full_model_path(req.model_path, diffusers_skip_ref_follow=True)
    else:
        path = Path(req.model_path)

    logger.warning(f"Deleting model '{path}' of type '{req.model_type}'")

    if not path.is_symlink():
        if not path.exists():
            websocket_manager.broadcast_sync(
                data=Notification(
                    severity="error",
                    message="Model not found",
                )
            )
            raise HTTPException(404, "Model not found")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        os.unlink(path)

    websocket_manager.broadcast_sync(data=Data(data_type="refresh_models", data={}))
    return {"message": "Model deleted"}


@router.post("/download-model")
def download_checkpoint(
    link: str, model_type: Literal["Checkpoint", "TextualInversion", "LORA", "VAE"]
) -> str:
    "Download a model from a link and return the path to the downloaded file."

    mtype = model_type.casefold()
    if mtype == "checkpoint":
        folder = "models"
    elif mtype == "textualinversion":
        folder = "textual-inversion"
    elif mtype == "lora":
        folder = "lora"
    elif mtype == "vae":
        folder = "vae"
    else:
        raise ValueError(f"Unknown model type {mtype}")

    saved_path = download_file(link, Path("data") / folder, True).as_posix()
    websocket_manager.broadcast_sync(Data(data_type="refresh_models", data={}))

    return saved_path
