from typing import Dict
import torch
from core.inference.pytorch import PyTorchInferenceModel
import gc
from core.types import SupportedModel, Txt2ImgQueueEntry
from core.inference.volta_accelerate import DemoDiffusion
import time


class ModelHandler:
    def __init__(self) -> None:
        self.generated_models: Dict[SupportedModel, Union[DemoDiffusion, PyTorchInferenceModel]] = dict()

    def generate(self, job: Txt2ImgQueueEntry):
        if job.model not in self.generated_models:
            print("Model not loaded")
            if job.backend == "TensorRT":
                print("Selecting TRT")
                print("Creating...")
                self.generated_models[job.model] = DemoDiffusion(
                    model_path=job.model.value,
                    denoising_steps=50,
                    denoising_fp16="fp16",
                    scheduler="LMSD",
                    hf_token="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",
                    verbose=False,
                    nvtx_profile=False,
                    max_batch_size=16
                )
                print("Loading engines...")
                self.generated_models[job.model].loadEngines(
                    engine_dir='engine/'+job.model.value,
                    onnx_dir='onnx',
                    onnx_opset=16,
                    opt_batch_size=len(job.data.prompt),
                    opt_image_height=job.data.height,
                    opt_image_width=job.data.width,
                )
                print("Loading modules")
                self.generated_models[job.model].loadModules()
                print("Loading done")
            else:
                print("Selecting PyTorch")
                start_time = time.time()
                self.generated_models[job.model] = PyTorchInferenceModel(job.model.value, job.scheduler)
                self.generated_models[job.model].optimize()
                print(f"Finished loading in {time.time() - start_time:.2f}s")
        
        print("Model loaded")
        if isinstance(self.generated_models[job.model], DemoDiffusion):
            pipeline_time, images = self.generated_models[job.model].infer(
                [job.data.prompt],
                [job.data.negative_prompt],
                job.data.height,
                job.data.width,
                guidance_scale=job.data.guidance_scale,
                verbose=False,
                seed=job.data.seed,
                output_dir='output',
                )
            print('Success')
            return [images[0]]
        else:
            return self.generated_models[job.model].generate(job.data)

    def unload(self, model: SupportedModel):
        if model in self.generated_models:
            self.generated_models[model].unload()
            self.generated_models.pop(model)
        self.free_memory()

    def unload_all(self):
        for model in self.generated_models:
            self.unload(model)

    def free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
