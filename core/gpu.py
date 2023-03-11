import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.convert.convert import load_pipeline_from_original_stable_diffusion_ckpt
from core.errors import DimensionError
from core.inference.aitemplate import AITemplateStableDiffusion
from core.inference.pytorch import PyTorchStableDiffusion
from core.png_metadata import save_images
from core.queue import Queue
from core.types import (
    AITemplateBuildRequest,
    BuildRequest,
    ControlNetQueueEntry,
    ImageVariationsQueueEntry,
    Img2ImgQueueEntry,
    InferenceBackend,
    InpaintQueueEntry,
    Job,
    Txt2ImgQueueEntry,
)
from core.utils import run_in_thread_async

if TYPE_CHECKING:
    from core.tensorrt.volta_accelerate import TRTModel

logger = logging.getLogger(__name__)


class GPU:
    "GPU with models attached to it."

    def __init__(self, torch_gpu_id: int) -> None:
        self.gpu_id = torch_gpu_id
        self.queue: Queue = Queue()
        self.loaded_models: Dict[
            str, Union["TRTModel", PyTorchStableDiffusion, "AITemplateStableDiffusion"]
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

    async def generate(
        self,
        job: Union[
            Txt2ImgQueueEntry,
            Img2ImgQueueEntry,
            InpaintQueueEntry,
            ImageVariationsQueueEntry,
            ControlNetQueueEntry,
        ],
    ):
        "Generate images from the queue"

        def generate_thread_call(job: Job) -> List[Image.Image]:
            model: Union[
                "TRTModel", PyTorchStableDiffusion, AITemplateStableDiffusion
            ] = self.loaded_models[job.model]

            shared.current_steps = job.data.steps * job.data.batch_count
            shared.current_done_steps = 0

            if isinstance(model, PyTorchStableDiffusion):
                logger.debug("Generating with PyTorch")
                images: List[Image.Image] = model.generate(job)
                self.memory_cleanup()
                return images
            elif isinstance(model, AITemplateStableDiffusion):
                logger.debug("Generating with AITemplate")
                images: List[Image.Image] = model.generate(job)
                self.memory_cleanup()
                return images
            else:
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
                    scheduler=job.data.scheduler,
                )
                self.memory_cleanup()
                return images

        # Check width and height passed by the user
        if not isinstance(job, ImageVariationsQueueEntry):
            if job.data.width % 8 != 0 or job.data.height % 8 != 0:
                raise DimensionError("Width and height must be divisible by 8")

        # Wait for turn
        await self.queue.wait_for_turn(job.data.id)

        start_time = time.time()

        try:
            images: Optional[List[Image.Image]]
            images = await run_in_thread_async(func=generate_thread_call, args=(job,))

            assert images is not None

            if job.save_image:
                save_images(images, job)
        except Exception as err:  # pylint: disable=broad-except
            self.queue.mark_finished()
            raise err

        deltatime = time.time() - start_time

        self.queue.mark_finished()

        return (images, deltatime)

    async def load_model(
        self,
        model: str,
        backend: InferenceBackend,
    ):
        "Load a model into memory"

        logger.debug(f"Loading {model} with {backend} backend")

        def load_model_thread_call(
            model: str,
            backend: InferenceBackend,
        ):
            if model in [self.loaded_models]:
                logger.debug(f"{model} is already loaded")
                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "Model already loaded",
                        f"{model} is already loaded with {'PyTorch' if isinstance(self.loaded_models[model], PyTorchStableDiffusion) else 'TensorRT'} backend",
                    )
                )
                return

            start_time = time.time()

            if backend == "TensorRT":
                logger.debug("Selecting TensorRT")

                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "TensorRT",
                        f"Loading {model} into memory, this may take a while",
                    )
                )

                from core.tensorrt.volta_accelerate import TRTModel

                trt_model = TRTModel(
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
                    enable_preview=True,
                    static_batch=True,
                    static_shape=True,
                )
                logger.debug("Loading modules")
                trt_model.loadModules()
                self.loaded_models[model] = trt_model
                logger.debug("Loading done")

            elif backend == "AITemplate":
                logger.debug("Selecting AITemplate")

                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "AITemplate",
                        f"Loading {model} into memory, this may take a while",
                    )
                )

                pt_model = AITemplateStableDiffusion(
                    model_id=model,
                    device=self.cuda_id,
                )
                self.loaded_models[model] = pt_model

            else:
                logger.debug("Selecting PyTorch")

                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "PyTorch",
                        f"Loading {model} into memory, this may take a while",
                    )
                )

                pt_model = PyTorchStableDiffusion(
                    model_id=model,
                    device=self.cuda_id,
                )
                self.loaded_models[model] = pt_model

            logger.info(f"Finished loading in {time.time() - start_time:.2f}s")

            websocket_manager.broadcast_sync(
                Notification(
                    "success",
                    "Model loaded",
                    f"{model} loaded with {backend} backend",
                )
            )

        await run_in_thread_async(func=load_model_thread_call, args=(model, backend))

    def loaded_models_list(self) -> list:
        "Return a list of loaded models"
        return list(self.loaded_models.keys())

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

            if isinstance(model, PyTorchStableDiffusion):
                logger.debug(f"Unloading PyTorch model: {model_type}")
                model.unload()
            elif isinstance(model, AITemplateStableDiffusion):
                logger.debug(f"Unloading AITemplate model: {model_type}")
                model.unload()
            else:
                from core.tensorrt.volta_accelerate import TRTModel

                assert isinstance(model, TRTModel)
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

        def convert_from_ckpt_thread_call(**kwargs):
            save_path = kwargs.pop("dump_path")
            pipe = load_pipeline_from_original_stable_diffusion_ckpt(**kwargs)
            pipe.save_pretrained(save_path, safe_serialization=True)

        await run_in_thread_async(
            func=convert_from_ckpt_thread_call,
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

    async def accelerate(self, model: str):
        "Convert a model to a TensorRT model"

        def trt_accelerate_thread_call():
            from core.tensorrt.volta_accelerate import TRTModel

            trt_model = TRTModel(
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

        await run_in_thread_async(func=trt_accelerate_thread_call)

    async def build_trt_engine(self, request: BuildRequest):
        "Build a TensorRT engine from a request"

        from .inference.tensorrt import TensorRTModel

        def trt_build_thread_call():
            model = TensorRTModel(model_id=request.model_id, use_f32=False)
            model.generate_engine(request=request)

        await run_in_thread_async(func=trt_build_thread_call)
        logger.info("TensorRT engine successfully built")

    async def build_aitemplate_engine(self, request: AITemplateBuildRequest):
        "Convert a model to a AITemplate engine"

        def ait_build_thread_call():
            from core.aitemplate.scripts.compile import compile_diffusers

            compile_diffusers(
                batch_size=request.batch_size,
                local_dir=request.model_id,
                height=request.height,
                width=request.width,
            )

        await run_in_thread_async(func=ait_build_thread_call)

        logger.info("AITemplate engine successfully built")

    async def to_fp16(self, model: str):
        "Convert a model to FP16"

        def model_to_f16_thread_call():
            pt_model = PyTorchStableDiffusion(
                model_id=model,
                device=self.cuda_id,
            )

            pt_model.save(path="converted/" + model.split("/")[-1])

        await run_in_thread_async(func=model_to_f16_thread_call)
