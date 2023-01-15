import gc
import logging
import os
import time
from typing import TYPE_CHECKING, Dict, List, Literal, Union

import torch
from PIL.Image import Image

from api import websocket_manager
from api.websockets import Notification
from core import shared
from core.errors import AutoLoadDisabledError
from core.functions import pytorch_callback
from core.inference.pytorch import PyTorchInferenceModel
from core.types import KDiffusionScheduler, Scheduler, Txt2ImgQueueEntry

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion

logger = logging.getLogger(__name__)


class ModelHandler:
    "Handles model loading and unloading"

    def __init__(self) -> None:
        self.generated_models: Dict[
            str, Union["DemoDiffusion", PyTorchInferenceModel]
        ] = {}

    def load_model(
        self,
        model: str,
        backend: Literal["PyTorch", "TensorRT"],
        device: str = "cuda",
    ):
        "Load a model into memory"

        if model in self.generated_models:
            logger.debug(f"{model} is already loaded")
            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "Model already loaded",
                    f"{model} is already loaded with {'PyTorch' if isinstance(self.generated_models[model], PyTorchInferenceModel) else 'TensorRT'} backend",
                )
            )
            return

        if backend == "TensorRT":

            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "TensorRT",
                    f"Loading {model} into memory, this may take a while",
                )
            )

            from core.inference.volta_accelerate import DemoDiffusion

            trt_model = DemoDiffusion(
                model_path=model,
                denoising_steps=25,
                denoising_fp16=True,
                hf_token=os.environ["HUGGINGFACE_TOKEN"],
                verbose=False,
                nvtx_profile=False,
                max_batch_size=16,
            )
            logger.debug("Loading engines...")
            trt_model.loadEngines(
                engine_dir="engine/" + model,
                onnx_dir="onnx",
                onnx_opset=16,
                opt_batch_size=150,
                opt_image_height=512,
                opt_image_width=512,
            )
            logger.debug("Loading modules")
            trt_model.loadModules()
            self.generated_models[model] = trt_model
            logger.debug("Loading done")
        else:
            logger.debug("Selecting PyTorch")

            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "PyTorch",
                    f"Loading {model} into memory, this may take a while",
                )
            )

            start_time = time.time()
            pt_model = PyTorchInferenceModel(
                model_id=model,
                scheduler=KDiffusionScheduler.euler_a,
                device=device,
                callback=pytorch_callback,
                callback_steps=1,
            )
            pt_model.optimize()
            self.generated_models[model] = pt_model
            logger.info(f"Finished loading in {time.time() - start_time:.2f}s")

        websocket_manager.broadcast_sync(
            Notification(
                "success",
                "Model loaded",
                f"{model} loaded with {'PyTorch' if backend == 'PyTorch' else 'TensorRT'} backend",
            )
        )

    def generate(self, job: Txt2ImgQueueEntry) -> List[Image]:
        "Generate an image(s) from a prompt"

        if job.model not in self.generated_models:
            if not job.autoload:
                websocket_manager.broadcast_sync(
                    Notification(
                        "error",
                        "Model not loaded",
                        "The model you are trying to use is not loaded, please load it first",
                    )
                )

                raise AutoLoadDisabledError

            self.load_model(model=job.model, backend=job.backend)

        model = self.generated_models[job.model]

        shared.current_model = model
        shared.current_steps = job.data.steps

        if isinstance(model, PyTorchInferenceModel):
            logger.debug("Generating with PyTorch")
            scheduler = job.scheduler
            assert isinstance(scheduler, KDiffusionScheduler)
            data = model.generate(
                job.data, scheduler=scheduler, use_karras_sigmas=job.use_karras_sigmas
            )
            self.free_memory()
            return data

        logger.debug("Generating with TensorRT")
        images: List[Image]

        scheduler = job.scheduler
        assert isinstance(scheduler, Scheduler)

        _, images = model.infer(
            [job.data.prompt],
            [job.data.negative_prompt],
            job.data.height,
            job.data.width,
            guidance_scale=job.data.guidance_scale,
            verbose=False,
            seed=job.data.seed,
            output_dir="output",
            num_of_infer_steps=job.data.steps,
            scheduler=scheduler,
        )
        self.free_memory()
        return [images[0]]

    def unload(self, model_type: str):
        "Unload a model from memory and free up GPU memory"

        if model_type in self.generated_models:
            model = self.generated_models[model_type]

            if isinstance(model, PyTorchInferenceModel):
                logger.debug("Unloading PyTorch model")
                model.unload()
            else:
                from core.inference.volta_accelerate import DemoDiffusion

                assert isinstance(model, DemoDiffusion)
                logger.debug("Unloading TensorRT model")
                model.teardown()

            self.generated_models.pop(model_type)
            logger.debug("Unloaded model")

        self.free_memory()
        logger.debug("Freed memory")

    def unload_all(self):
        "Unload all models from memory and free up GPU memory"

        logger.debug("Unloading all models")

        for model in self.generated_models:
            self.unload(model)

    def free_memory(self):
        "Free up GPU memory by purging dangling objects"

        torch.cuda.empty_cache()
        logger.debug("Cache emptied")

        torch.cuda.ipc_collect()
        logger.debug("IPC collected")

        gc.collect()
        logger.debug("Garbage collector cleaned")
