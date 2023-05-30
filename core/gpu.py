import asyncio
import logging
import math
import multiprocessing
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import torch
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.config import config
from core.errors import DimensionError, InferenceInterruptedError, ModelNotLoadedError
from core.flags import HighResFixFlag
from core.inference.aitemplate import AITemplateStableDiffusion
from core.inference.functions import download_model
from core.inference.pytorch import PyTorchSDUpscaler, PyTorchStableDiffusion
from core.inference.real_esrgan import RealESRGAN
from core.interrogation.base_interrogator import InterrogationResult
from core.png_metadata import save_images
from core.queue import Queue
from core.types import (
    AITemplateBuildRequest,
    AITemplateDynamicBuildRequest,
    ControlNetQueueEntry,
    Img2ImgQueueEntry,
    InferenceBackend,
    InpaintQueueEntry,
    InterrogatorQueueEntry,
    Job,
    LoraLoadRequest,
    ONNXBuildRequest,
    SDUpscaleQueueEntry,
    TextualInversionLoadRequest,
    TRTBuildRequest,
    Txt2ImgQueueEntry,
    UpscaleQueueEntry,
)
from core.utils import image_grid

if TYPE_CHECKING:
    from core.inference.onnx_sd import OnnxStableDiffusion
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
                PyTorchSDUpscaler,
                "OnnxStableDiffusion",
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
            SDUpscaleQueueEntry,
        ],
    ):
        "Generate images from the queue"

        def generate_thread_call(job: Job) -> List[Image.Image]:
            try:
                model: Union[
                    "TensorRTModel",
                    PyTorchStableDiffusion,
                    AITemplateStableDiffusion,
                    PyTorchSDUpscaler,
                    "OnnxStableDiffusion",
                ] = self.loaded_models[job.model]
            except KeyError as err:
                websocket_manager.broadcast_sync(
                    Notification(
                        "error",
                        "Model not loaded",
                        f"Model {job.model} is not loaded, please load it first",
                    )
                )

                logger.debug("Model not loaded on any GPU. Raising error")
                raise ModelNotLoadedError(f"Model {job.model} is not loaded") from err

            shared.interrupt = False

            if job.flags:
                logger.debug(f"Job flags: {job.flags}")

            steps = job.data.steps

            strength: float = getattr(job.data, "strength", 1.0)
            steps = math.floor(steps * strength)

            extra_steps: int = 0
            if "highres_fix" in job.flags:
                flag = HighResFixFlag.from_dict(job.flags["highres_fix"])
                extra_steps = math.floor(flag.steps * flag.strength)

            shared.current_steps = steps * job.data.batch_count + extra_steps
            shared.current_done_steps = 0

            if not isinstance(job, ControlNetQueueEntry):
                from core import shared_dependent

                if shared_dependent.cached_controlnet_preprocessor is not None:
                    # Wipe cached controlnet preprocessor
                    shared_dependent.cached_controlnet_preprocessor = None
                    self.memory_cleanup()

            if isinstance(model, PyTorchStableDiffusion):
                logger.debug("Generating with PyTorch")
                images: List[Image.Image] = model.generate(job)
            elif isinstance(model, AITemplateStableDiffusion):
                logger.debug("Generating with AITemplate")
                images: List[Image.Image] = model.generate(job)
            elif isinstance(model, PyTorchSDUpscaler):
                logger.debug("Generating with PyTorchSDUpscaler")
                images: List[Image.Image] = model.generate(job)
            else:
                from core.inference.onnx_sd import OnnxStableDiffusion

                if isinstance(model, OnnxStableDiffusion):
                    logger.debug("Generating with ONNX")
                    images: List[Image.Image] = model.generate(job)
                else:
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
                # if config.api.clear_memory_policy == "always":
                #     self.memory_cleanup()
                # return images

            self.memory_cleanup()
            return images

        try:
            # Check width and height passed by the user
            if not isinstance(
                job,
                SDUpscaleQueueEntry,
            ):
                if job.data.width % 8 != 0 or job.data.height % 8 != 0:
                    raise DimensionError("Width and height must be divisible by 8")

            # Wait for turn in the queue
            await self.queue.wait_for_turn(job.data.id)

            start_time = time.time()

            # Generate images
            try:
                generated_images = await asyncio.to_thread(generate_thread_call, job)

                assert generated_images is not None

                # [pre, out...]
                images = generated_images
                grid = image_grid(images)

                # Save only if needed
                if job.save_image:
                    if isinstance(job, ControlNetQueueEntry):
                        if not job.data.save_preprocessed:  # type: ignore
                            # Save only the output
                            preprocessed = images[0]
                            images = images[1:]

                            out = save_images(images, job)
                            images = [preprocessed, *images]
                        else:
                            out = save_images(images, job)
                    else:
                        out = save_images(images, job)

                    # URL Strings returned from R2 bucket if saved there
                    if out:
                        logger.debug(f"Strings returned from R2: {len(out)}")
                        images = out

            except Exception as err:  # pylint: disable=broad-except
                self.memory_cleanup()
                await self.queue.mark_finished(job.data.id)
                raise err

            deltatime = time.time() - start_time

            # Mark job as finished, so the next job can start
            self.memory_cleanup()
            await self.queue.mark_finished(job.data.id)

            # Check if user wants the preprocessed image back (ControlNet only)
            if isinstance(job, ControlNetQueueEntry):
                if not job.data.return_preprocessed:  # type: ignore
                    # Remove the preprocessed image
                    images = images[1:]

            # Append grid to the list of images if needed
            if isinstance(images[0], Image.Image) and len(images) > 1:
                images = [grid, *images]

            return (images, deltatime)
        except InferenceInterruptedError:
            if config.frontend.on_change_timer == 0:
                await websocket_manager.broadcast(
                    Notification(
                        "warning",
                        "Inference interrupted",
                        "The inference was forcefully interrupted",
                    )
                )
            return ([], 0.0)

        except Exception as e:  # pylint: disable=broad-except
            if not isinstance(e, ModelNotLoadedError):
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
            elif backend == "ONNX":
                logger.debug("Selecting ONNX")

                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "ONNX",
                        f"Loading {model} into memory, this may take a while",
                    )
                )

                from core.inference.onnx_sd import OnnxStableDiffusion

                pt_model = OnnxStableDiffusion(model_id=model)
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

                if model in ["stabilityai/stable-diffusion-x4-upscaler"]:
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

        await asyncio.to_thread(load_model_thread_call, model, backend)

    def loaded_models_list(self) -> list:
        "Return a list of loaded models"
        return list(self.loaded_models.keys())

    def memory_cleanup(self):
        "Release all unused memory"
        if config.api.clear_memory_policy == "always":
            if torch.cuda.is_available():
                logger.debug(f"Cleaning up GPU memory: {self.gpu_id}")

                with torch.cuda.device(self.gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    async def unload(self, model_type: str):
        "Unload a model from memory and free up GPU memory"

        def unload_thread_call(model_type: str):
            if model_type in self.loaded_models:
                model = self.loaded_models[model_type]

                if hasattr(model, "unload"):
                    logger.debug(f"Unloading model: {model_type}")
                    model.unload()
                else:
                    from core.tensorrt.volta_accelerate import TRTModel

                    assert isinstance(
                        model, TRTModel
                    ), "Model is not a TRTModel and does not have an unload method"
                    logger.debug(f"Unloading TensorRT model: {model_type}")
                    model.teardown()

                del self.loaded_models[model_type]
                self.memory_cleanup()
                logger.debug("Unloaded model")

        await asyncio.to_thread(unload_thread_call, model_type)

    async def unload_all(self):
        "Unload all models from memory and free up GPU memory"

        logger.debug("Unloading all models")

        for model in list(self.loaded_models.keys()):
            await self.unload(model)

        self.memory_cleanup()

    async def build_trt_engine(self, request: TRTBuildRequest):
        "Build a TensorRT engine from a request"

        from .inference.tensorrt import TensorRTModel

        logger.debug(f"Building engine for {request.model_id}...")

        def trt_build_thread_call():
            model = TensorRTModel(model_id=request.model_id, use_f32=False)
            model.generate_engine(request=request)

        await asyncio.to_thread(trt_build_thread_call)
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

        await asyncio.to_thread(ait_build_thread_call)

        logger.debug(f"AI Template built for {request.model_id}.")

        logger.info("AITemplate engine successfully built")

    async def build_dynamic_aitemplate_engine(
        self, request: AITemplateDynamicBuildRequest
    ):
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
            from core.aitemplate.dynamic_compile import compile_diffusers

            compile_diffusers(
                batch_size=request.batch_size,
                local_dir_or_id=request.model_id,
                height=request.height,
                width=request.width,
                clip_chunks=request.clip_chunks,
            )

            self.memory_cleanup()

        await asyncio.to_thread(ait_build_thread_call)

        logger.debug(f"AI Template built for {request.model_id}.")

        logger.info("AITemplate engine successfully built")

    async def build_onnx_engine(self, request: ONNXBuildRequest):
        "Convert a model to a ONNX engine"

        from core.inference.onnx_sd import OnnxStableDiffusion

        logger.debug(f"Building ONNX for {request.model_id}...")

        def onnx_build_thread_call():
            pipe = OnnxStableDiffusion(
                model_id=request.model_id,
                autoload=False,
            )

            pipe.convert_pytorch_to_onnx(
                device=config.api.device,
                model_id=request.model_id,
                simplify_unet=request.simplify_unet,
                convert_to_fp16=request.convert_to_fp16,
                target=request.quant_dict,
            )

            self.memory_cleanup()

        await asyncio.to_thread(onnx_build_thread_call)

        logger.info(f"ONNX engine successfully built for {request.model_id}.")

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

        await asyncio.to_thread(model_to_f16_thread_call)

        logger.debug(f"Converted {model}.")

    async def download_huggingface_model(self, model: str):
        "Download a model from the internet."

        await asyncio.to_thread(download_model, model)

    async def load_lora(self, req: LoraLoadRequest):
        "Inject a Lora model into a model"

        if req.model in self.loaded_models:
            internal_model = self.loaded_models[req.model]

            if isinstance(internal_model, PyTorchStableDiffusion):
                logger.info(
                    f"Loading Lora model: {req.lora}, weights: ({req.unet_weight}, {req.text_encoder_weight})"
                )

                internal_model.load_lora(
                    req.lora, req.unet_weight, req.text_encoder_weight
                )

                websocket_manager.broadcast_sync(
                    Notification(
                        "success",
                        "Lora model loaded",
                        f"Lora model {req.lora} loaded",
                    )
                )

        else:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not found",
                    f"Model {req.model} not found",
                )
            )
            logger.error(f"Model {req.model} not found")

    async def load_textual_inversion(self, req: TextualInversionLoadRequest):
        "Inject a textual inversion model into a model"

        if req.model in self.loaded_models:
            internal_model = self.loaded_models[req.model]

            if isinstance(internal_model, PyTorchStableDiffusion):
                logger.info(f"Loading textual inversion model: {req.textual_inversion}")

                internal_model.load_textual_inversion(req.textual_inversion)

                websocket_manager.broadcast_sync(
                    Notification(
                        "success",
                        "Textual inversion model loaded",
                        f"Textual inversion model {req.textual_inversion} loaded",
                    )
                )

        else:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not found",
                    f"Model {req.model} not found",
                )
            )
            logger.error(f"Model {req.model} not found")

    async def interrogate(self, job: InterrogatorQueueEntry):
        "Generate captions for image"

        def generate_call(job: InterrogatorQueueEntry):
            if job.model == "deepdanbooru":
                from core.interrogation.deepdanbooru import DeepdanbooruInterrogator

                model = DeepdanbooruInterrogator(
                    device=config.api.device, autoload=True
                )
                out = model.generate(job)
                model.unload()
                return out

            elif job.model == "clip":
                from core.interrogation.clip import CLIPInterrogator

                model = CLIPInterrogator(device=config.api.device, autoload=True)
                out = model.generate(job)
                model.unload()
                return out

            elif job.model == "flamingo":
                from core.interrogation.flamingo import FlamingoInterrogator

                model = FlamingoInterrogator(device=config.api.device)
                out = model.generate(job)
                model.unload()
                return out
            else:
                raise ValueError(f"Model {job.model} not implemented")

        output: InterrogationResult = await asyncio.to_thread(generate_call, job)
        return output

    async def upscale(self, job: UpscaleQueueEntry):
        "Upscale an image by a specified factor"

        def generate_call(job: UpscaleQueueEntry):
            if job.model in [
                "RealESRGAN_x4plus",
                "RealESRNet_x4plus",
                "RealESRGAN_x4plus_anime_6B",
                "RealESRGAN_x2plus",
                "RealESR-general-x4v3",
            ]:
                t = time.time()
                pipe = RealESRGAN(
                    model_name=job.model,
                    tile=job.data.tile_size,
                    tile_pad=job.data.tile_padding,
                )

                image = pipe.generate(job)
                pipe.unload()
                deltatime = time.time() - t
                return image, deltatime
            else:
                raise ValueError(f"Model {job.model} not implemented")

        image: Image.Image
        time_: float
        image, time_ = await asyncio.to_thread(generate_call, job)

        save_images([image], job)

        return image, time_
