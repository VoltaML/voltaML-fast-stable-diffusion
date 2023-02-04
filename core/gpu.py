import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import torch
from diffusers.schedulers import KarrasDiffusionSchedulers
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.convert.convert import load_pipeline_from_original_stable_diffusion_ckpt
from core.errors import DimensionError
from core.inference.pytorch import PyTorchInferenceModel
from core.png_metadata import save_images
from core.queue import Queue
from core.types import Img2ImgQueueEntry, Txt2ImgQueueEntry
from core.utils import run_in_thread_async

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion

logger = logging.getLogger(__name__)


class GPU:
    "GPU with models attached to it."

    def __init__(self, torch_gpu_id: int) -> None:
        self.gpu_id = torch_gpu_id
        self.queue: Queue = Queue()
        self.loaded_models: Dict[
            str, Union["DemoDiffusion", PyTorchInferenceModel]
        ] = {}

    @property
    def cuda_id(self) -> str:
        "Returns the CUDA ID of the GPU."
        return f"cuda:{self.gpu_id}"

    def vram_free(self) -> float:
        "Returns the amount of free VRAM on the GPU in MB."
        return (
            torch.cuda.get_device_properties(self.gpu_id).total_memory
            - torch.cuda.memory_allocated(self.gpu_id)
        ) / 1024**2

    def vram_used(self) -> float:
        "Returns the amount of used VRAM on the GPU in MB."
        return torch.cuda.memory_allocated(self.gpu_id) / 1024**2

    async def generate(self, job: Union[Txt2ImgQueueEntry, Img2ImgQueueEntry]):
        "Generate images from the queue"

        logging.info(f"Adding job {job.data.id} to queue")

        if job.data.width % 8 != 0 or job.data.height % 8 != 0:
            raise DimensionError("Width and height must be divisible by 8")

        await self.queue.wait_for_turn(job.data.id)

        start_time = time.time()

        try:
            if isinstance(job, Txt2ImgQueueEntry):
                images = await self.txt2img(job)
            elif isinstance(job, Img2ImgQueueEntry):
                images = await self.img2img(job)
        except Exception as err:  # pylint: disable=broad-except
            self.queue.mark_finished()
            raise err

        deltatime = time.time() - start_time

        self.queue.mark_finished()

        return (images, deltatime)

    async def load_model(
        self,
        model: str,
        backend: Literal["PyTorch", "TensorRT"],
    ):
        "Load a model into memory"

        logger.debug(f"Loading {model} with {backend} backend")

        def thread_call(
            model: str,
            backend: Literal["PyTorch", "TensorRT"],
        ):
            if model in [self.loaded_models]:
                logger.debug(f"{model} is already loaded")
                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "Model already loaded",
                        f"{model} is already loaded with {'PyTorch' if isinstance(self.loaded_models[model], PyTorchInferenceModel) else 'TensorRT'} backend",
                    )
                )
                return

            if backend == "TensorRT":
                logger.debug("Selecting TensorRT")

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
                    max_batch_size=9,
                )
                logger.debug("Loading engines...")
                trt_model.loadEngines(
                    engine_dir="engine/" + model,
                    onnx_dir="onnx",
                    onnx_opset=16,
                    opt_batch_size=1,
                    opt_image_height=512,
                    opt_image_width=512,
                )
                logger.debug("Loading modules")
                trt_model.loadModules()
                self.loaded_models[model] = trt_model
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
                    device=self.cuda_id,
                    callback_steps=shared.image_decode_steps,
                )
                pt_model.optimize()
                self.loaded_models[model] = pt_model
                logger.info(f"Finished loading in {time.time() - start_time:.2f}s")

            websocket_manager.broadcast_sync(
                Notification(
                    "success",
                    "Model loaded",
                    f"{model} loaded with {'PyTorch' if backend == 'PyTorch' else 'TensorRT'} backend",
                )
            )

        _, err = await run_in_thread_async(func=thread_call, args=(model, backend))
        if err:
            raise err

    def loaded_models_list(self) -> list:
        "Return a list of loaded models"
        return list(self.loaded_models.keys())

    async def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Generate an image(s) from a prompt"

        def thread_call(job: Txt2ImgQueueEntry) -> List[Image.Image]:
            model: Union["DemoDiffusion", PyTorchInferenceModel] = self.loaded_models[
                job.model
            ]

            shared.current_model = model
            shared.current_steps = (
                job.data.steps * job.data.batch_count * job.data.batch_size
            )

            if isinstance(model, PyTorchInferenceModel):
                logger.debug("Generating with PyTorch")
                images: List[Image.Image] = model.txt2img(job)
                self.memory_cleanup()
                return images

            logger.debug("Generating with TensorRT")
            images: List[Image.Image]

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
                scheduler=job.scheduler
                if job.scheduler
                else KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler,
            )
            self.memory_cleanup()
            return images

        images: Optional[List[Image.Image]]
        error: Optional[Exception]
        images, error = await run_in_thread_async(func=thread_call, args=(job,))

        if error is not None:
            raise error

        assert images is not None

        if job.save_image:
            save_images(images, job)

        return images

    async def img2img(self, job: Img2ImgQueueEntry) -> List[Image.Image]:
        "Run an image2image job"

        def thread_call(job: Img2ImgQueueEntry) -> List[Image.Image]:
            model: Union["DemoDiffusion", PyTorchInferenceModel] = self.loaded_models[
                job.model
            ]

            shared.current_model = model
            shared.current_steps = (
                job.data.steps * job.data.batch_count * job.data.batch_size
            )

            if isinstance(model, PyTorchInferenceModel):
                logger.debug("Generating with PyTorch")
                images: List[Image.Image] = model.img2img(job)
                self.memory_cleanup()
                return images

            logger.debug("Generating with TensorRT")
            images: List[Image.Image]

            _, images = model.infer(
                [job.data.prompt],
                [job.data.negative_prompt],
                job.data.height,
                job.data.width,
                guidance_scale=job.data.guidance_scale,
                verbose=False,
                output_dir="output",
                num_of_infer_steps=job.data.steps,
                scheduler=job.scheduler,
            )
            self.memory_cleanup()
            return images

        images: Optional[List[Image.Image]]
        error: Optional[Exception]
        images, error = await run_in_thread_async(func=thread_call, args=(job,))

        if error is not None:
            raise error

        assert images is not None

        if job.save_image:
            save_images(images, job)

        return images

    def memory_cleanup(self):
        "Release all unused memory"

        if torch.cuda.is_available():
            logger.debug(f"Cleaning up GPU memory: {self.gpu_id}")

            with torch.cuda.device(self.gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    async def unload(self, model_type: str):
        "Unload a model from memory and free up GPU memory"

        if model_type in self.loaded_models:
            model = self.loaded_models[model_type]

            if isinstance(model, PyTorchInferenceModel):
                logger.debug(f"Unloading PyTorch model: {model_type}")
                model.unload()
            else:
                from core.inference.volta_accelerate import DemoDiffusion

                assert isinstance(model, DemoDiffusion)
                logger.debug(f"Unloading TensorRT model: {model_type}")
                model.teardown()

            del self.loaded_models[model_type]
            self.memory_cleanup()
            logger.debug("Unloaded model")

    async def unload_all(self):
        "Unload all models from memory and free up GPU memory"

        logger.debug("Unloading all models")

        for model in list(self.loaded_models.keys()):
            await self.unload(model)

        self.memory_cleanup()

    async def convert_from_checkpoint(self, checkpoint: str, is_sd2: bool):
        "Convert a checkpoint to a proper model structure that can be loaded"

        from_safetensors = ".safetensors" in checkpoint

        def call(**kwargs):
            save_path = kwargs.pop("dump_path")
            pipe = load_pipeline_from_original_stable_diffusion_ckpt(**kwargs)
            pipe.save_pretrained(save_path, safe_serialization=True)

        _, err = await run_in_thread_async(
            func=call,
            kwarkgs={
                "checkpoint_path": checkpoint,
                "device": self.cuda_id,
                "extract_ema": True,
                "from_safetensors": from_safetensors,
                "original_config_file": "v1-inference.yaml",
                "upcast_attention": is_sd2,
                "image_size": 768 if is_sd2 else 512,
                "dump_path": f"converted/{Path(checkpoint).name}",
            },
        )

        if err is not None:
            raise err

    async def accelerate(self, model: str):
        "Convert a model to a TensorRT model"

        def call():
            from core.inference.volta_accelerate import DemoDiffusion

            trt_model = DemoDiffusion(
                model_path=model,
                denoising_steps=25,
                denoising_fp16=True,
                hf_token=os.environ["HUGGINGFACE_TOKEN"],
                verbose=False,
                nvtx_profile=False,
                max_batch_size=9,
            )

            trt_model.loadEngines(
                engine_dir="engine/" + model,
                onnx_dir="onnx",
                onnx_opset=16,
                opt_batch_size=1,
                opt_image_height=512,
                opt_image_width=512,
                force_build=True,
                static_batch=True,
                static_shape=True,
            )
            trt_model.teardown()
            del trt_model

        _, err = await run_in_thread_async(func=call)
        if err is not None:
            raise err
