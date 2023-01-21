import logging
from typing import Dict, List, Literal, Optional, Union

import torch

from api import websocket_manager
from api.websockets.notification import Notification
from core.errors import ModelNotLoadedError
from core.gpu import GPU
from core.types import Img2ImgQueueEntry, Txt2ImgQueueEntry

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

    async def txt2img(self, job: Txt2ImgQueueEntry):
        "Send a text to image job to the cluster."

        raise NotImplementedError

    async def img2img(self):
        "Send an image to image job to the cluster."

        raise NotImplementedError

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

    async def loaded_models(self) -> Dict[int, List[str]]:
        "Return a list of all loaded models."

        models: Dict = {}
        for gpu in self.gpus:
            models[gpu.gpu_id] = list(gpu.loaded_models.keys())

        return models

    async def generate(self, job: Union[Txt2ImgQueueEntry, Img2ImgQueueEntry]):
        "Generate images from the queue"

        # Find gpu with the model loaded, raise error if not found
        # if multiple found, use one that has the least jobs in queue

        useful_gpus: List[GPU] = []
        for gpu in self.gpus:
            if job.model in gpu.loaded_models.keys():
                useful_gpus.append(gpu)

        if len(useful_gpus) == 0:
            raise ModelNotLoadedError("Model not loaded on any GPU.")

        best_gpu: GPU = useful_gpus[0]
        for gpu in useful_gpus:
            if best_gpu is None:
                best_gpu = gpu
                continue

            if len(best_gpu.queue.jobs) > len(gpu.queue.jobs):
                best_gpu = gpu

        return await best_gpu.generate(job)
