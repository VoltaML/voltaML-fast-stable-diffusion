import logging
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.config import config
from core.errors import DimensionError, InferenceInterruptedError, ModelNotLoadedError
from core.inference.aitemplate import AITemplateStableDiffusion
from core.inference.functions import download_model
from core.inference.pytorch import PyTorchStableDiffusion
from core.inference.pytorch_upscale import PyTorchSDUpscaler
from core.inference.real_esrgan import RealESRGAN
from core.png_metadata import save_images
from core.queue import Queue
from core.types import (
    AITemplateBuildRequest,
    BuildRequest,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    InferenceBackend,
    InpaintQueueEntry,
    Job,
    RealESRGANQueueEntry,
    SDUpscaleQueueEntry,
    Txt2ImgQueueEntry,
)
from core.utils import run_in_thread_async

if TYPE_CHECKING:
    from core.inference.tensorrt import TensorRTModel

logger = logging.getLogger(__name__)


class GPU:
    "GPU with models attached to it."

    def __init__(self, torch_gpu_id: int) -> None:
        self.gpu_id = torch_gpu_id
        self.queue: Queue = Queue()
        self.loaded_models: Dict[
            str,
            Union[
                "TensorRTModel",
                PyTorchStableDiffusion,
                "AITemplateStableDiffusion",
                "RealESRGAN",
                PyTorchSDUpscaler,
            ],
        ] = {}

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
            ControlNetQueueEntry,
            RealESRGANQueueEntry,
            SDUpscaleQueueEntry,
        ],
    ):
        "Generate images from the queue"

        def generate_thread_call(job: Job) -> List[Image.Image]:
            model: Union[
                "TensorRTModel",
                PyTorchStableDiffusion,
                AITemplateStableDiffusion,
                RealESRGAN,
                PyTorchSDUpscaler,
            ] = self.loaded_models[job.model]

            if not isinstance(job, RealESRGANQueueEntry):
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
            elif isinstance(model, PyTorchSDUpscaler):
                logger.debug("Generating with PyTorchSDUpscaler")
                images: List[Image.Image] = model.generate(job)
                self.memory_cleanup()
                return images
            elif isinstance(model, RealESRGAN):
                logger.debug("Generating with RealESRGAN")
                images: List[Image.Image] = model.generate(job)
                self.memory_cleanup()
                return images
            else:
                assert not isinstance(job, RealESRGANQueueEntry)

                raise NotImplementedError("TensorRT is not supported at the moment")

                # logger.debug("Generating with TensorRT")
                # images: List[Image.Image]

                # _, images = model.infer(
                #     [job.data.prompt],
                #     [job.data.negative_prompt],
                #     job.data.height,
                #     job.data.width,
                #     guidance_scale=job.data.guidance_scale,
                #     verbose=False,
                #     seed=job.data.seed,
                #     output_dir="output",
                #     num_of_infer_steps=job.data.steps,
                #     scheduler=job.data.scheduler,
                # )
                # self.memory_cleanup()
                # return images

        try:
            # Check width and height passed by the user
            if not isinstance(
                job,
                (RealESRGANQueueEntry, SDUpscaleQueueEntry),
            ):
                if job.data.width % 8 != 0 or job.data.height % 8 != 0:
                    raise DimensionError("Width and height must be divisible by 8")

            # Wait for turn
            await self.queue.wait_for_turn(job.data.id)

            start_time = time.time()

            try:
                images: Optional[List[Image.Image]]
                images = await run_in_thread_async(
                    func=generate_thread_call, args=(job,)
                )

                assert images is not None

                if job.save_image:
                    save_images(images, job)
            except Exception as err:  # pylint: disable=broad-except
                self.queue.mark_finished()
                raise err

            deltatime = time.time() - start_time

            self.queue.mark_finished()

            return (images, deltatime)
        except InferenceInterruptedError:
            await websocket_manager.broadcast(
                Notification(
                    "warning",
                    "Inference interrupted",
                    "The inference was forcefully interrupted",
                )
            )
            return ([], 0.0)

        except ValueError as err:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not loaded",
                    "The model you are trying to use is not loaded, please load it first",
                )
            )

            logger.debug("Model not loaded on any GPU. Raising error")
            raise ModelNotLoadedError("Model not loaded on any GPU") from err

        except Exception as e:  # pylint: disable=broad-except
            await websocket_manager.broadcast(
                Notification(
                    "error",
                    "Inference error",
                    f"An error occurred: {type(e).__name__} - {e}",
                )
            )

            raise e

    async def load_model(
        self,
        model: str,
        backend: InferenceBackend,
    ):
        "Load a model into memory"

        if model in self.loaded_models:
            logger.debug(f"{model} is already loaded")
            websocket_manager.broadcast_sync(
                Notification(
                    "info",
                    "Model already loaded",
                    f"{model} is already loaded with {'PyTorch' if isinstance(self.loaded_models[model], PyTorchStableDiffusion) else 'TensorRT'} backend",
                )
            )
            return

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

                from core.inference.tensorrt import TensorRTModel

                trt_model = TensorRTModel(
                    model_id=model,
                )
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
                    device=config.api.device,
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

                if model in [
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "RealESR-general-x4v3",
                ]:
                    pt_model = RealESRGAN(
                        model_name=model,
                    )
                elif model in ["stabilityai/stable-diffusion-x4-upscaler"]:
                    pt_model = PyTorchSDUpscaler()

                else:
                    pt_model = PyTorchStableDiffusion(
                        model_id=model,
                        device=config.api.device,
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

    async def build_trt_engine(self, request: BuildRequest):
        "Build a TensorRT engine from a request"

        from .inference.tensorrt import TensorRTModel

        logger.debug(f"Building engine for {request.model_id}...")

        def trt_build_thread_call():
            model = TensorRTModel(model_id=request.model_id, use_f32=False)
            model.generate_engine(request=request)

        await run_in_thread_async(func=trt_build_thread_call)
        logger.info("TensorRT engine successfully built")

    async def build_aitemplate_engine(self, request: AITemplateBuildRequest):
        "Convert a model to a AITemplate engine"

        logger.debug(f"Building AI Template for {request.model_id}...")

        # Set the number of threads to use and keep them within boundaries of the system
        if request.threads:
            if request.threads < 1:
                request.threads = 1
            elif request.threads > multiprocessing.cpu_count():
                request.threads = multiprocessing.cpu_count()
            else:
                config.aitemplate.num_threads = request.threads

        def ait_build_thread_call():
            from core.aitemplate.compile import compile_diffusers

            compile_diffusers(
                batch_size=request.batch_size,
                local_dir_or_id=request.model_id,
                height=request.height,
                width=request.width,
            )

            self.memory_cleanup()

        await run_in_thread_async(func=ait_build_thread_call)

        logger.debug(f"AI Template built for {request.model_id}.")

        logger.info("AITemplate engine successfully built")

    async def convert_model(self, model: str, safetensors: bool = False):
        "Convert a model to FP16"

        logger.debug(f"Converting {model}...")

        def model_to_f16_thread_call():
            pt_model = PyTorchStableDiffusion(
                model_id=model,
                device=config.api.device,
                autoload=True,
            )

            model_name = model.split("/")[-1]
            model_name = model_name.split(".")[0]

            pt_model.save(
                path=str(Path("data/models").joinpath(model_name)),
                safetensors=safetensors,
            )
            pt_model.unload()

        await run_in_thread_async(func=model_to_f16_thread_call)

        logger.debug(f"Converted {model}.")

    async def download_huggingface_model(self, model: str):
        "Download a model from the internet."

        await run_in_thread_async(download_model, args=(model,))

    async def load_lora(self, model: str, lora: str):
        "Inject a Lora model into a model"

        if model in self.loaded_models:
            internal_model = self.loaded_models[model]

            if isinstance(internal_model, PyTorchStableDiffusion):
                logger.debug(f"Loading Lora model: {lora}")

                internal_model.load_lora(lora)

                websocket_manager.broadcast_sync(
                    Notification(
                        "success",
                        "Lora model loaded",
                        f"Lora model {lora} loaded",
                    )
                )

        else:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not found",
                    f"Model {model} not found",
                )
            )
            logger.error(f"Model {model} not found")
