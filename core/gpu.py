import asyncio
import logging
import math
import multiprocessing
import time
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import torch
from diffusers.utils import is_xformers_available
from packaging import version
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.config import config
from core.errors import InferenceInterruptedError, ModelNotLoadedError
from core.inference.ait import AITemplateStableDiffusion
from core.inference.esrgan import RealESRGAN, Upscaler
from core.inference.functions import is_ipex_available
from core.inference.pytorch import PyTorchStableDiffusion
from core.inference.sdxl import SDXLStableDiffusion
from core.interrogation.base_interrogator import InterrogationResult
from core.optimizations import is_hypertile_available
from core.png_metadata import save_images
from core.queue import Queue
from core.types import (
    AITemplateBuildRequest,
    AITemplateDynamicBuildRequest,
    Capabilities,
    ControlNetQueueEntry,
    InferenceBackend,
    InferenceJob,
    InterrogatorQueueEntry,
    Job,
    ONNXBuildRequest,
    TextualInversionLoadRequest,
    UpscaleQueueEntry,
    VaeLoadRequest,
)
from core.utils import convert_to_image, image_grid, preprocess_job

if TYPE_CHECKING:
    from core.inference.onnx import OnnxStableDiffusion

logger = logging.getLogger(__name__)


class GPU:
    "GPU with models attached to it."

    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.loaded_models: Dict[
            str,
            Union[
                PyTorchStableDiffusion,
                "AITemplateStableDiffusion",
                "OnnxStableDiffusion",
            ],
        ] = {}
        self.capabilities = self._get_capabilities()

    def _get_capabilities(self) -> Capabilities:
        "Returns all of the capabilities of this GPU."
        cap = Capabilities()

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            cap.supported_self_attentions = [
                ["SDP Attention", "sdpa"],
                *cap.supported_self_attentions,
            ]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cap.supported_backends.append(
                    [f"(CUDA) {torch.cuda.get_device_name(i)}", f"cuda:{i}"]
                )
        try:
            # Not find_spec because we actually do end up using it
            import torch_directml

            for i in range(torch_directml.device_count()):
                cap.supported_backends.append(
                    [f"(DML) {torch_directml.device_name(i)}", f"privateuseone:{i}"]
                )
        except ImportError:
            pass
        if torch.backends.mps.is_available():  # type: ignore
            cap.supported_backends.append(["MPS", "mps"])
        if torch.is_vulkan_available():
            cap.supported_backends.append(["Vulkan", "vulkan"])
        if is_ipex_available():
            cap.supported_backends.append(["Intel CPU/GPU", "xpu"])

        test_suite = ["float16", "bfloat16"]
        support_map: Dict[str, List[str]] = {}
        for device in [torch.device("cpu"), torch.device(config.api.device)]:
            support_map[device.type] = []
            for dt in test_suite:
                dtype = getattr(torch, dt)
                a = torch.tensor([1.0], device=device, dtype=dtype)
                b = torch.tensor([2.0], device=device, dtype=dtype)
                try:
                    torch.matmul(a, b)
                    support_map[device.type].append(dt)
                except RuntimeError:
                    pass
        for t, s in support_map.items():
            if t == "cpu":
                cap.supported_precisions_cpu = ["float32"] + s
            else:
                cap.supported_precisions_gpu = ["float32"] + s
        try:
            cap.supported_torch_compile_backends = (
                torch._dynamo.list_backends()  # type: ignore
            )
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                import bitsandbytes.functional as F

                a = torch.tensor([1.0]).cuda()
                b, b_state = F.quantize_fp4(torch.tensor([2.0]).cuda())
                bnb.matmul_4bit(a, b, quant_state=b_state)  # type: ignore
                cap.supports_int8 = True
                logger.debug("GPU supports int8")
            except Exception:
                logger.debug("GPU does not support int8")

        cap.supports_xformers = is_xformers_available()
        cap.supports_triton = find_spec("triton") is not None

        if find_spec("flash_attn") is not None:
            cap.supported_self_attentions.append(["Flash-Attention", "flash-attn"])

        if cap.supports_xformers:
            cap.supported_self_attentions = [
                ["xFormers", "xformers"],
                *cap.supported_self_attentions,
            ]

        if torch.cuda.is_available():
            caps = torch.cuda.get_device_capability(0)
            if caps[0] < 7:
                cap.has_tensor_cores = False

            if caps[0] == 8 and caps[1] >= 6:
                cap.has_tensorfloat = True
            elif caps[0] >= 9:
                cap.has_tensorfloat = True

        if is_hypertile_available():
            cap.hypertile_available = True

        return cap

    def vram_free(self) -> float:
        "Returns the amount of free VRAM on the GPU in MB."
        index = torch.device(config.api.device).index
        return (
            torch.cuda.get_device_properties(index).total_memory
            - torch.cuda.memory_allocated(index)
        ) / 1024**2

    def vram_used(self) -> float:
        "Returns the amount of used VRAM on the GPU in MB."
        index = torch.device(config.api.device).index
        return torch.cuda.memory_allocated(index) / 1024**2

    async def generate(
        self,
        job: InferenceJob,
    ):
        "Generate images from the queue"

        job = preprocess_job(job)

        def generate_thread_call(job: Job) -> List[Image.Image]:
            try:
                model: Union[
                    PyTorchStableDiffusion,
                    AITemplateStableDiffusion,
                    SDXLStableDiffusion,
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

            shared.current_done_steps = 0

            if not isinstance(job, ControlNetQueueEntry):
                from core import shared_dependent

                if shared_dependent.cached_controlnet_preprocessor is not None:
                    # Wipe cached controlnet preprocessor
                    shared_dependent.cached_controlnet_preprocessor = None
                    self.memory_cleanup()

            # shared.current_model = model

            if isinstance(model, PyTorchStableDiffusion):
                logger.debug("Generating with PyTorch")
                images: List[Image.Image] = model.generate(job)
            elif isinstance(model, SDXLStableDiffusion):
                logger.debug("Generating with SDXL (PyTorch)")
                images: List[Image.Image] = model.generate(job)
            elif isinstance(model, AITemplateStableDiffusion):
                logger.debug("Generating with AITemplate")
                images: List[Image.Image] = model.generate(job)
            else:
                from core.inference.onnx import OnnxStableDiffusion

                if isinstance(model, OnnxStableDiffusion):
                    logger.debug("Generating with ONNX")
                    images: List[Image.Image] = model.generate(job)
                else:
                    raise NotImplementedError("Unknown model type")

            self.memory_cleanup()
            return images

        try:
            # Wait for turn in the queue
            await self.queue.wait_for_turn(job.data.id)

            start_time = time.time()

            # Generate images
            try:
                generated_images = await asyncio.to_thread(generate_thread_call, job)

                assert generated_images is not None

                # [pre, out...]
                images = generated_images

                if not config.api.disable_grid:
                    grid = image_grid(images)
                else:
                    grid = None

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

            except Exception as err:
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
                images = [grid, *images] if grid else images

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

        except Exception as e:
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
                    f"{model} is already loaded",
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
                        f"{model} is already loaded",
                    )
                )
                return

            start_time = time.time()

            if backend == "AITemplate":
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

                from core.inference.onnx import OnnxStableDiffusion

                pt_model = OnnxStableDiffusion(model_id=model)
                self.loaded_models[model] = pt_model
            elif backend == "SDXL":
                logger.debug("Selecting SDXL")

                websocket_manager.broadcast_sync(
                    Notification(
                        "info",
                        "SDXL",
                        f"Loading {model} into memory, this may take a while",
                    )
                )

                sdxl_model = SDXLStableDiffusion(
                    model_id=model,
                    device=config.api.device,
                )
                self.loaded_models[model] = sdxl_model
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
                index = torch.device(config.api.device).index
                logger.debug(f"Cleaning up GPU memory: {index}")

                with torch.cuda.device(index):
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
            from core.aitemplate.compile import compile_diffusers

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

        from core.inference.onnx import OnnxStableDiffusion

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
                model_id=model, device=config.api.device, autoload=True, bare=True
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

        from diffusers import DiffusionPipeline

        await asyncio.to_thread(DiffusionPipeline.download, model, resume_download=True)

    async def load_vae(self, req: VaeLoadRequest):
        "Change the models VAE"

        if req.model in self.loaded_models:
            internal_model = self.loaded_models[req.model]

            if hasattr(internal_model, "change_vae"):
                logger.info(f"Loading VAE model: {req.vae}")

                internal_model.change_vae(req.vae)  # type: ignore

                websocket_manager.broadcast_sync(
                    Notification(
                        "success",
                        "VAE model loaded",
                        f"VAE model {req.vae} loaded",
                    )
                )
        else:
            websocket_manager.broadcast_sync(
                Notification("error", "Model not found", f"Model {req.model} not found")
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
            t: float = time.time()

            if "realesr" in job.model.lower():
                pipe = RealESRGAN(
                    model_name=job.model,
                    tile=job.data.tile_size,
                    tile_pad=job.data.tile_padding,
                )

                image = pipe.generate(job)
                pipe.unload()

            else:
                pipe = Upscaler(
                    model=job.model,
                    device_id=torch.device(config.api.device).index,
                    cpu=config.api.device == "cpu",
                    fp16=True,
                )

                input_image = convert_to_image(job.data.image)
                image = pipe.run(input_img=input_image, scale=job.data.upscale_factor)[
                    0
                ]

            deltatime = time.time() - t
            return image, deltatime

        image: Image.Image
        time_: float
        image, time_ = await asyncio.to_thread(generate_call, job)

        save_images([image], job)

        return image, time_
