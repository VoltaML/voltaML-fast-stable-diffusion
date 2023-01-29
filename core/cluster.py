import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.diffusers.functions import download_model
from core.errors import InferenceInterruptedError, ModelNotLoadedError
from core.gpu import GPU
from core.types import Img2ImgQueueEntry, Txt2ImgQueueEntry
from core.utils import run_in_thread_async

logger = logging.getLogger(__name__)


class Cluster:
    "Represents a cluster of GPUs with models attached to them."

    def __init__(self) -> None:
        self.gpus: List[GPU] = []
        self.populate_gpus()

    def populate_gpus(self) -> None:
        "Find all available GPUs and add them to the cluster."

        for i in range(torch.cuda.device_count()):
            if i not in [i.gpu_id for i in self.gpus]:
                self.gpus.append(GPU(i))

    async def least_loaded_gpu(self) -> GPU:
        "Return the GPU with the most free memory in the system."

        return sorted(self.gpus, key=lambda x: x.vram_free(), reverse=True)[0]

    async def load_model(
        self,
        model: str,
        backend: Literal["PyTorch", "TensorRT"],
        preferred_gpu: Optional[int] = None,
    ):
        "Load a model onto the cluster. Chooses the least loaded GPU by default."

        gpu: GPU

        if preferred_gpu is not None:
            if preferred_gpu > torch.cuda.device_count() - 1:
                raise ValueError("Invalid GPU ID.")

            gpu: GPU = [i for i in self.gpus if i.gpu_id == preferred_gpu][0]
        else:
            logger.debug("Finding least loaded GPU...")
            gpu = await self.least_loaded_gpu()
            logger.debug(f"Least loaded GPU is: {gpu.gpu_id}.")

        if not model in gpu.loaded_models:
            logger.debug(f"Loading {model} on GPU {gpu.gpu_id}...")
            await gpu.load_model(model=model, backend=backend)
        else:
            await websocket_manager.broadcast(
                Notification(
                    "warning",
                    title="Model already loaded",
                    message=f"{model} is already loaded on selected GPU.",
                )
            )

    async def download_model(self, model: str):
        "Download a model from the internet."

        _, err = await run_in_thread_async(download_model, args=(model,))

        if err:
            raise err

    async def loaded_models(self) -> Dict[int, List[str]]:
        "Return a list of all loaded models."

        models: Dict = {}
        for gpu in self.gpus:
            models[gpu.gpu_id] = list(gpu.loaded_models.keys())

        return models

    async def generate(
        self, job: Union[Txt2ImgQueueEntry, Img2ImgQueueEntry]
    ) -> Tuple[List[Image.Image], float]:
        "Generate images from the queue"

        # Find gpu with the model loaded, raise error if not found
        # if multiple found, use one that has the least jobs in queue

        shared.interrupt = False

        useful_gpus: List[GPU] = []
        for gpu in self.gpus:
            if job.model in gpu.loaded_models.keys():
                useful_gpus.append(gpu)

        if len(useful_gpus) == 0:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not loaded",
                    "The model you are trying to use is not loaded, please load it first",
                )
            )

            raise ModelNotLoadedError("Model not loaded on any GPU.")

        best_gpu: GPU = useful_gpus[0]
        for gpu in useful_gpus:
            if best_gpu is None:
                best_gpu = gpu
                continue

            if len(best_gpu.queue.jobs) > len(gpu.queue.jobs):
                best_gpu = gpu

        try:
            return await best_gpu.generate(job)
        except InferenceInterruptedError:
            await websocket_manager.broadcast(
                Notification(
                    "warning",
                    "Inference interrupted",
                    "The inference was forcefully interrupted",
                )
            )

            return ([], 0.0)

    async def convert_from_checkpoint(self, checkpoint_path: str, is_sd2: bool):
        "Convert a checkpoint to a proper model structure that can be loaded"

        best_gpu: GPU = self.gpus[0]
        for gpu in self.gpus:
            if best_gpu is None:
                best_gpu = gpu
                continue

            if len(best_gpu.queue.jobs) > len(gpu.queue.jobs):
                best_gpu = gpu

        await best_gpu.convert_from_checkpoint(checkpoint_path, is_sd2=is_sd2)
