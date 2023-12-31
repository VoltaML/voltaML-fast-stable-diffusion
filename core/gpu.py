import logging
import math
import multiprocessing
import time
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import torch
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from PIL import Image

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared
from core.config import config
from core.errors import InferenceInterruptedError, ModelNotLoadedError
from core.flags import ADetailerFlag, HighResFixFlag, UpscaleFlag
from core.inference.ait import AITemplateStableDiffusion
from core.inference.esrgan import RealESRGAN, Upscaler
from core.inference.functions import is_ipex_available
from core.inference.pytorch import PyTorchStableDiffusion
from core.inference.sdxl import SDXLStableDiffusion
from core.inference.utilities.latents import scale_latents
from core.interrogation.base_interrogator import InterrogationResult
from core.optimizations import is_hypertile_available
from core.png_metadata import save_images
from core.queue import Queue
from core.types import (
    ADetailerQueueEntry,
    AITemplateBuildRequest,
    AITemplateDynamicBuildRequest,
    Capabilities,
    ControlNetQueueEntry,
    Img2imgData,
    Img2ImgQueueEntry,
    InferenceBackend,
    InferenceJob,
    InpaintData,
    InpaintQueueEntry,
    InterrogatorQueueEntry,
    Job,
    ONNXBuildRequest,
    PyTorchModelBase,
    TextualInversionLoadRequest,
    Txt2ImgQueueEntry,
    UpscaleData,
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
                "SDXLStableDiffusion",
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
                try:
                    dtype = getattr(torch, dt)
                    a = torch.tensor([1.0], device=device, dtype=dtype)
                    b = torch.tensor([2.0], device=device, dtype=dtype)
                    torch.matmul(a, b)
                    support_map[device.type].append(dt)
                except RuntimeError:
                    pass
                except AssertionError:
                    pass
        for t, s in support_map.items():
            if t == "cpu":
                cap.supported_precisions_cpu = (
                    ["float32"] + s + ["float8_e4m3fn", "float8_e5m2"]
                )
            else:
                cap.supported_precisions_gpu = (
                    ["float32"] + s + ["float8_e4m3fn", "float8_e5m2"]
                )
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

    def highres_flag(
        self, job: Job, images: Union[List[Image.Image], torch.Tensor]
    ) -> List[Image.Image]:
        flag = job.flags["highres_fix"]
        flag = HighResFixFlag.from_dict(flag)

        if flag.mode == "latent":
            assert isinstance(images, (torch.Tensor, torch.FloatTensor))
            latents = images

            latents = scale_latents(
                latents=latents,
                scale=flag.scale,
                latent_scale_mode=flag.latent_scale_mode,
            )

            height = latents.shape[2] * 8
            width = latents.shape[3] * 8
            output_images = latents
        else:
            from core.shared_dependent import gpu

            assert isinstance(images, List)
            output_images = []

            for image in images:
                output: tuple[Image.Image, float] = gpu.upscale(
                    UpscaleQueueEntry(
                        data=UpscaleData(
                            id=job.data.id,
                            # FastAPI validation error, we need to do this so that we can pass in a PIL image
                            image=image,  # type: ignore
                            upscale_factor=flag.scale,
                        ),
                        model=flag.image_upscaler,
                        save_image=False,
                    )
                )
                output_images.append(output[0])

            output_images = output_images[0]  # type: ignore
            height = int(flag.scale * job.data.height)
            width = int(flag.scale * job.data.width)

        data = Img2imgData(
            prompt=job.data.prompt,
            negative_prompt=job.data.negative_prompt,
            image=output_images,  # type: ignore
            scheduler=job.data.scheduler,
            batch_count=job.data.batch_count,
            batch_size=job.data.batch_size,
            strength=flag.strength,
            steps=flag.steps,
            guidance_scale=job.data.guidance_scale,
            prompt_to_prompt_settings=job.data.prompt_to_prompt_settings,
            seed=job.data.seed,
            self_attention_scale=job.data.self_attention_scale,
            sigmas=job.data.sigmas,
            sampler_settings=job.data.sampler_settings,
            height=height,
            width=width,
        )

        img2img_job = Img2ImgQueueEntry(
            data=data,
            model=job.model,
        )

        result: List[Image.Image] = self.run_inference(img2img_job)
        return result

    def upscale_flag(self, job: Job, images: List[Image.Image]) -> List[Image.Image]:
        logger.debug("Upscaling image")

        flag = UpscaleFlag(**job.flags["upscale"])

        final_images = []
        for image in images:
            upscale_job = UpscaleQueueEntry(
                data=UpscaleData(
                    image=image,  # type: ignore # Pydantic would cry if we extend the union
                    upscale_factor=flag.upscale_factor,
                    tile_padding=flag.tile_padding,
                    tile_size=flag.tile_size,
                ),
                model=flag.model,
            )

            final_images.append(self.upscale(upscale_job)[0])

        return final_images

    def adetailer_flag(self, job: Job, images: List[Image.Image]) -> List[Image.Image]:
        logger.debug("Running ADetailer")

        flag = ADetailerFlag(**job.flags["adetailer"])
        data = asdict(flag)
        mask_blur = data.pop("mask_blur")
        mask_dilation = data.pop("mask_dilation")
        mask_padding = data.pop("mask_padding")
        iterations = data.pop("iterations")
        upscale = data.pop("upscale")
        data.pop("enabled", None)

        data["prompt"] = job.data.prompt
        data["negative_prompt"] = job.data.negative_prompt
        data["scheduler"] = data.pop("sampler")

        data = InpaintData(**data)

        assert data is not None

        final_images = []
        for image in images:
            data.image = image  # type: ignore
            data.prompt = job.data.prompt
            data.negative_prompt = job.data.negative_prompt

            adetailer_job = ADetailerQueueEntry(
                data=data,
                mask_blur=mask_blur,
                mask_dilation=mask_dilation,
                mask_padding=mask_padding,
                iterations=iterations,
                upscale=upscale,
                model=job.model,
            )

            final_images.extend(self.run_inference(adetailer_job))

        return final_images

    def postprocess(
        self, job: Job, images: Union[List[Image.Image], torch.Tensor]
    ) -> List[Image.Image]:
        "Postprocess images"

        logger.debug(f"Postprocessing flags: {job.flags}")

        if "highres_fix" in job.flags:
            images = self.highres_flag(job, images)

        if "adetailer" in job.flags:
            assert isinstance(images, list)
            images = self.adetailer_flag(job, images)

        if "upscale" in job.flags:
            assert isinstance(images, list)
            images = self.upscale_flag(job, images)

        assert isinstance(images, list)
        return images

    def set_callback_target(self, job: Job):
        "Set the callback target for the job, updates the shared object and also returns the target"

        if isinstance(job, Txt2ImgQueueEntry):
            target = "txt2img"
        elif isinstance(job, Img2ImgQueueEntry):
            target = "img2img"
        elif isinstance(job, ControlNetQueueEntry):
            target = "controlnet"
        elif isinstance(job, InpaintQueueEntry):
            target = "inpainting"
        else:
            raise ValueError("Unknown job type")

        shared.current_method = target
        return target

    def run_inference(self, job: Job) -> List[Image.Image]:
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

        if isinstance(model, PyTorchStableDiffusion):
            logger.debug("Generating with SD PyTorch")
            shared.current_model = "SD1.x"
            images: Union[List[Image.Image], torch.Tensor] = model.generate(job)
        elif isinstance(model, SDXLStableDiffusion):
            logger.debug("Generating with SDXL (PyTorch)")
            shared.current_model = "SDXL"
            images: Union[List[Image.Image], torch.Tensor] = model.generate(job)
        elif isinstance(model, AITemplateStableDiffusion):
            logger.debug("Generating with SD AITemplate")
            images: Union[List[Image.Image], torch.Tensor] = model.generate(job)
        else:
            from core.inference.onnx import OnnxStableDiffusion

            if isinstance(model, OnnxStableDiffusion):
                logger.debug("Generating with SD ONNX")
                images: Union[List[Image.Image], torch.Tensor] = model.generate(job)
            else:
                raise NotImplementedError("Unknown model type")

        self.memory_cleanup()

        # Run postprocessing
        images = self.postprocess(job, images)
        return images

    def generate(
        self,
        job: InferenceJob,
    ):
        "Generate images from the queue"

        job = preprocess_job(job)

        try:
            # Wait for turn in the queue
            self.queue.wait_for_turn(job.data.id)

            start_time = time.time()

            # Generate images
            try:
                self.set_callback_target(job)
                generated_images = self.run_inference(job)

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
                self.queue.mark_finished(job.data.id)
                raise err

            deltatime = time.time() - start_time

            # Mark job as finished, so the next job can start
            self.memory_cleanup()
            self.queue.mark_finished(job.data.id)

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
                websocket_manager.broadcast_sync(
                    Notification(
                        "warning",
                        "Inference interrupted",
                        "The inference was forcefully interrupted",
                    )
                )
            return ([], 0.0)

        except Exception as e:
            if not isinstance(e, ModelNotLoadedError):
                websocket_manager.broadcast_sync(
                    Notification(
                        "error",
                        "Inference error",
                        f"An error occurred: {type(e).__name__} - {e}",
                    )
                )

            raise e

    def load_model(
        self,
        model: str,
        backend: InferenceBackend,
        type: PyTorchModelBase,
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
            elif type == "SDXL":
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

        load_model_thread_call(model, backend)

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

    def unload(self, model_type: str):
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
            else:
                raise ValueError(f"Model {model_type} not loaded")

        unload_thread_call(model_type)

    def unload_all(self):
        "Unload all models from memory and free up GPU memory"

        logger.debug("Unloading all models")

        for model in list(self.loaded_models.keys()):
            self.unload(model)

        self.memory_cleanup()

    def build_aitemplate_engine(self, request: AITemplateBuildRequest):
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

        ait_build_thread_call()

        logger.debug(f"AI Template built for {request.model_id}.")
        logger.info("AITemplate engine successfully built")

    def build_dynamic_aitemplate_engine(self, request: AITemplateDynamicBuildRequest):
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

        ait_build_thread_call()

        logger.debug(f"AI Template built for {request.model_id}.")

        logger.info("AITemplate engine successfully built")

    def build_onnx_engine(self, request: ONNXBuildRequest):
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

        onnx_build_thread_call()

        logger.info(f"ONNX engine successfully built for {request.model_id}.")

    def convert_model(self, model: str, safetensors: bool = False):
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

        model_to_f16_thread_call()

        logger.debug(f"Converted {model}.")

    def download_huggingface_model(self, model: str):
        "Download a model from the internet."

        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

        DiffusionPipeline.download(model, resume_download=True)

    def load_vae(self, req: VaeLoadRequest):
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

    def load_textual_inversion(self, req: TextualInversionLoadRequest):
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
            if isinstance(internal_model, SDXLStableDiffusion):
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
                logger.warning(f"Model {req.model} does not support textual inversion")

        else:
            websocket_manager.broadcast_sync(
                Notification(
                    "error",
                    "Model not found",
                    f"Model {req.model} not found",
                )
            )
            logger.error(f"Model {req.model} not found")

    def interrogate(self, job: InterrogatorQueueEntry):
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

        output: InterrogationResult = generate_call(job)
        return output

    def upscale(self, job: UpscaleQueueEntry):
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
        image, time_ = generate_call(job)

        save_images([image], job)

        return image, time_
