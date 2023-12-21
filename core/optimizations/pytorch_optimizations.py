import logging
from typing import Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

from core.config import config

from .attn import set_attention_processor
from .compile.trace_utils import trace_ipex, trace_model
from .dtype import cast
from .offload import set_offload
from .upcast import upcast_vae

logger = logging.getLogger(__name__)


USE_DISK_OFFLOAD = False


def optimize_model(
    pipe: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    device,
    is_for_aitemplate: bool = False,
    silent: bool = False,
) -> None:
    "Optimize the model for inference."

    logger.disabled = silent

    # Tuple[Supported, Enabled by default, Enabled]
    hardware_scheduling = experimental_check_hardware_scheduling()

    if hardware_scheduling[2] == 1 and not is_for_aitemplate:
        logger.warning(
            "Hardware accelerated scheduling is turned on! This will have a HUGE negative impact on performance"
        )
        if hardware_scheduling[1] == 1:
            logger.warning(
                "You most likely didn't even know this was turned on. Windows 11 enables it by default on NVIDIA devices"
            )
        logger.warning(
            "You can read about this issue on https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/3889"
        )
        logger.warning(
            'You can disable it by going inside Graphics Settings → "Default Graphics Settings" and disabling "Hardware-accelerated GPU Scheduling"'
        )

    offload = (
        config.api.offload != "disabled"
        and is_pytorch_pipe(pipe)
        and not is_for_aitemplate
    )
    can_offload = (
        any(map(lambda x: x not in config.api.device, ["cpu", "vulkan", "mps"]))
        and offload
    )

    pipe = cast(pipe, device, config.api.dtype, can_offload)

    if "cuda" in config.api.device and not is_for_aitemplate:
        supports_tf = supports_tf32(device)
        if config.api.reduced_precision:
            if supports_tf:
                torch.set_float32_matmul_precision("medium")
                logger.info("Optimization: Enabled all reduced precision operations")
            else:
                logger.warning(
                    "Optimization: Device capability is not higher than 8.0, skipping most of reduction"
                )
                logger.info(
                    "Optimization: Reduced precision operations enabled (fp16 only)"
                )
        torch.backends.cuda.matmul.allow_tf32 = config.api.reduced_precision and supports_tf  # type: ignore
        torch.backends.cudnn.allow_tf32 = config.api.reduced_precision and supports_tf  # type: ignore
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.api.reduced_precision  # type: ignore
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.api.reduced_precision and supports_tf  # type: ignore

        logger.info(
            f"Optimization: CUDNN {'' if config.api.deterministic_generation else 'not '}using deterministic functions"
        )
        torch.backends.cudnn.deterministic = config.api.deterministic_generation  # type: ignore

        if config.api.cudnn_benchmark:
            logger.info("Optimization: CUDNN benchmark enabled")
        torch.backends.cudnn.benchmark = config.api.cudnn_benchmark  # type: ignore

    if is_pytorch_pipe(pipe):
        pipe.vae = optimize_vae(pipe.vae)

    # Attention slicing that should save VRAM (but is slower)
    slicing = config.api.attention_slicing
    if slicing != "disabled" and is_pytorch_pipe(pipe) and not is_for_aitemplate:
        if slicing == "auto":
            pipe.enable_attention_slicing()
            logger.info("Optimization: Enabled attention slicing")
        else:
            pipe.enable_attention_slicing(slicing)
            logger.info(f"Optimization: Enabled attention slicing ({slicing})")

    # xFormers and SPDA
    if not is_for_aitemplate:
        set_attention_processor(pipe)

        if config.api.autocast:
            logger.info("Optimization: Enabled autocast")

    if can_offload:
        # Offload to CPU

        for model_name in [
            "text_encoder",
            "text_encoder_2",
            "unet",
            "vae",
        ]:
            cpu_offloaded_model = getattr(pipe, model_name, None)
            if cpu_offloaded_model is not None:
                cpu_offloaded_model = set_offload(cpu_offloaded_model, device)
                setattr(pipe, model_name, cpu_offloaded_model)
        logger.info("Optimization: Offloaded model parts to CPU.")

    if config.api.free_u:
        pipe.enable_freeu(
            s1=config.api.free_u_s1,
            s2=config.api.free_u_s2,
            b1=config.api.free_u_b1,
            b2=config.api.free_u_b2,
        )

    if config.api.use_tomesd and not is_for_aitemplate:
        try:
            import tomesd

            tomesd.apply_patch(pipe.unet, ratio=config.api.tomesd_ratio, max_downsample=config.api.tomesd_downsample_layers)  # type: ignore
            logger.info("Optimization: Patched UNet for ToMeSD")
        except ImportError:
            logger.info(
                "Optimization: ToMeSD patch failed, despite having it enabled. Please check installation"
            )

    ipexed = False
    if "cpu" in config.api.device:
        from cpufeature import CPUFeature as cpu

        n = (cpu["num_virtual_cores"] // 4) * 3
        torch.set_num_threads(n)
        torch.set_num_interop_threads(n)

        logger.info(
            f"Running on an {cpu['VendorId']} device. Used threads: {torch.get_num_threads()}-{torch.get_num_interop_threads()} / {cpu['num_virtual_cores']}"
        )

        pipe.unet, ipexed = trace_ipex(pipe.unet, config.api.load_dtype, device, cpu)

    if config.api.trace_model and not ipexed and not is_for_aitemplate:
        logger.info("Optimization: Tracing model.")
        logger.warning("This will break controlnet and loras!")
        pipe.unet = trace_model(pipe.unet, config.api.load_dtype, device)  # type: ignore

    if config.api.torch_compile and not is_for_aitemplate:
        if config.api.attention_processor == "xformers":
            logger.warning(
                "Skipping torchscript compilation because xformers used for attention processor. Please change to SDPA to enable torchscript compilation."
            )
        else:
            logger.info(
                "Optimization: Compiling model with: %s",
                {
                    "fullgraph": config.api.torch_compile_fullgraph,
                    "dynamic": config.api.torch_compile_dynamic,
                    "backend": config.api.torch_compile_backend,
                    "mode": config.api.torch_compile_mode,
                },
            )


def supports_tf32(device: Optional[torch.device] = None) -> bool:
    "Checks if device is post-Ampere."

    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def experimental_check_hardware_scheduling() -> Tuple[int, int, int]:
    "When on windows check if user has hardware scheduling turned on"
    import sys

    if sys.platform != "win32":
        return (-1, -1, -1)

    import ctypes
    from pathlib import Path

    hardware_schedule_test = ctypes.WinDLL(
        str(Path() / "libs" / "hardware-accel-test.dll")
    ).HwSchEnabled
    hardware_schedule_test.restype = ctypes.POINTER(ctypes.c_int * 3)
    return [x for x in hardware_schedule_test().contents]  # type: ignore


def is_pytorch_pipe(pipe):
    "Checks if the pipe is a pytorch pipe"

    from .context_manager import InferenceContext

    return issubclass(pipe.__class__, (DiffusionPipeline, InferenceContext))


def optimize_vae(vae):
    "Optimize a VAE according to config defined in data/settings.json"
    vae = upcast_vae(vae)

    if hasattr(vae, "enable_slicing") and config.api.vae_slicing:
        vae.enable_slicing()
        logger.info("Optimization: Enabled VAE slicing")

    if config.api.vae_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        logger.info("Optimization: Enabled VAE tiling")
    return vae
