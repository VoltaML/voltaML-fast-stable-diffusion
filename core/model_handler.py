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
from core.types import SupportedModel, Txt2ImgQueueEntry

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion


class ModelHandler:
    "Handles model loading and unloading"

    def __init__(self) -> None:
        self.generated_models: Dict[
            SupportedModel, Union["DemoDiffusion", PyTorchInferenceModel]
        ] = {}

    def load_model(
        self,
        model: SupportedModel,
        backend: Literal["PyTorch", "TensorRT"],
        device: str = "cuda",
    ):
        "Load a model into memory"

        if model in self.generated_models:
            logging.debug(f"{model.value} is already loaded")
            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "Model already loaded",
                    f"{model.value} is already loaded with {'PyTorch' if isinstance(self.generated_models[model], PyTorchInferenceModel) else 'TensorRT'} backend",
                )
            )
            return

        if backend == "TensorRT":

            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "TensorRT",
                    f"Loading {model.value} into memory, this may take a while",
                )
            )

            from core.inference.volta_accelerate import DemoDiffusion

            trt_model = DemoDiffusion(
                model_path=model.value,
                denoising_steps=25,
                denoising_fp16=True,
                hf_token=os.environ["HUGGINGFACE_TOKEN"],
                verbose=False,
                nvtx_profile=False,
                max_batch_size=16,
            )
            logging.debug("Loading engines...")
            trt_model.loadEngines(
                engine_dir="engine/" + model.value,
                onnx_dir="onnx",
                onnx_opset=16,
                opt_batch_size=150,
                opt_image_height=512,
                opt_image_width=512,
            )
            logging.debug("Loading modules")
            trt_model.loadModules()
            self.generated_models[model] = trt_model
            logging.debug("Loading done")
        else:
            logging.debug("Selecting PyTorch")

            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "PyTorch",
                    f"Loading {model.value} into memory, this may take a while",
                )
            )

            start_time = time.time()
            pt_model = PyTorchInferenceModel(
                model.value,
                model.value,
                device=device,
                callback=pytorch_callback,
                callback_steps=1,
            )
            pt_model.optimize()
            self.generated_models[model] = pt_model
            logging.info(f"Finished loading in {time.time() - start_time:.2f}s")

        websocket_manager.broadcast_sync(
            Notification(
                "success",
                "Model loaded",
                f"{model.value} loaded with {'PyTorch' if backend == 'PyTorch' else 'TensorRT'} backend",
            )
        )

    def generate(self, job: Txt2ImgQueueEntry) -> List[Image]:
        "Generate an image(s) from a prompt"

        if job.model not in self.generated_models:
            if not job.autoload:
                raise AutoLoadDisabledError

            self.load_model(model=job.model, backend=job.backend)

        print("Model loaded")
        model = self.generated_models[job.model]

        shared.current_model = model
        shared.current_steps = job.data.steps

        if isinstance(model, PyTorchInferenceModel):
            data = model.generate(job.data, scheduler=job.scheduler)
            self.free_memory()
            return data
        else:
            images: List[Image]
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
                scheduler=job.scheduler,
            )
            self.free_memory()  # ! Might cause issues with TRT, need to check
            return [images[0]]

    def unload(self, model_type: SupportedModel):
        "Unload a model from memory and free up GPU memory"

        if model_type in self.generated_models:
            model = self.generated_models[model_type]

            if isinstance(model, PyTorchInferenceModel):
                model.unload()
            else:
                from core.inference.volta_accelerate import DemoDiffusion

                assert isinstance(model, DemoDiffusion)
                model.teardown()

            self.generated_models.pop(model_type)
        self.free_memory()

    def unload_all(self):
        "Unload all models from memory and free up GPU memory"

        for model in self.generated_models:
            self.unload(model)

    def free_memory(self):
        "Free up GPU memory by purging dangling objects"

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
