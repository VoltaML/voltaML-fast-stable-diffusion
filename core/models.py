import gc
import os
import time
from typing import Dict, List, Union

import torch
from PIL.Image import Image

from core.inference.pytorch import PyTorchInferenceModel
from core.inference.volta_accelerate import DemoDiffusion
from core.types import SupportedModel, Txt2ImgQueueEntry


class ModelHandler:
    "Handles model loading and unloading"

    def __init__(self) -> None:
        self.generated_models: Dict[
            SupportedModel, Union[DemoDiffusion, PyTorchInferenceModel]
        ] = {}

    def generate(self, job: Txt2ImgQueueEntry) -> List[Image]:
        "Generate an image(s) from a prompt"

        if job.model not in self.generated_models:
            print("Model not loaded")
            if job.backend == "TensorRT":
                print("Selecting TRT")
                print("Creating...")
                trt_model = DemoDiffusion(
                    model_path=job.model.value,
                    denoising_steps=50,
                    denoising_fp16=True,
                    scheduler="LMSD",
                    hf_token=os.environ["HUGGINGFACE_TOKEN"],
                    verbose=False,
                    nvtx_profile=False,
                    max_batch_size=16,
                )
                print("Loading engines...")
                trt_model.load_engines(
                    engine_dir="engine/" + job.model.value,
                    onnx_dir="onnx",
                    onnx_opset=16,
                    opt_batch_size=len(job.data.prompt),
                    opt_image_height=job.data.height,
                    opt_image_width=job.data.width,
                )
                print("Loading modules")
                trt_model.loadModules()
                self.generated_models[job.model] = trt_model
                print("Loading done")
            else:
                print("Selecting PyTorch")
                start_time = time.time()
                pt_model = PyTorchInferenceModel(job.model.value, job.scheduler)
                pt_model.optimize()
                self.generated_models[job.model] = pt_model
                print(f"Finished loading in {time.time() - start_time:.2f}s")

        print("Model loaded")
        trt_model = self.generated_models[job.model]
        if isinstance(trt_model, DemoDiffusion):
            images: List[Image]
            _, images = trt_model.infer(
                [job.data.prompt],
                [job.data.negative_prompt],
                job.data.height,
                job.data.width,
                guidance_scale=job.data.guidance_scale,
                verbose=False,
                seed=job.data.seed,
                output_dir="output",
            )
            print("Success")
            return [images[0]]
        else:
            return trt_model.generate(job.data)

    def unload(self, model: SupportedModel):
        "Unload a model from memory and free up GPU memory"

        if model in self.generated_models:
            if isinstance(model, DemoDiffusion):
                model.teardown()
            else:
                assert isinstance(model, PyTorchInferenceModel)
                model.unload()
            self.generated_models.pop(model)
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
