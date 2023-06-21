import logging
from typing import Optional, Tuple, Union

import torch
from cpufeature import CPUFeature as cpu
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.utils import is_accelerate_available
from packaging import version
from rich.console import Console

from core.config import config
from core.files import get_full_model_path
from core.inference.functions import torch_older_than_200

from .attn import set_attention_processor
from .iree import convert_pipe_state_to_iree
from .trace_utils import generate_inputs, trace_model

logger = logging.getLogger(__name__)
console = Console()


USE_DISK_OFFLOAD = False


def optimize_model(
    pipe: Union[StableDiffusionPipeline, StableDiffusionUpscalePipeline],
    device,
    is_for_aitemplate: bool = False,
) -> None:
    "Optimize the model for inference"
    from core.inference.functions import is_ipex_available

    with console.status("[bold green]Optimizing model..."):
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
            config.api.offload
            if (is_pytorch_pipe(pipe) and not is_for_aitemplate)
            else None
        )
        can_offload = config.api.device_type not in [
            "cpu",
            "iree",
            "vulkan",
            "mps",
        ] and (offload != "disabled" and offload is not None)

        # Took me an hour to understand why CPU stopped working...
        # Turns out AMD just lacks support for BF16...
        # Not mad, not mad at all... to be fair, I'm just disappointed
        if not can_offload and not is_for_aitemplate:
            pipe.to(device, torch_dtype=config.api.dtype)

        if config.api.device_type == "cuda" and not is_for_aitemplate:
            supports_tf = supports_tf32(device)
            if config.api.reduced_precision:
                if supports_tf:
                    logger.info(
                        "Optimization: Enabled all reduced precision operations"
                    )
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

        # Attention slicing that should save VRAM (but is slower)
        slicing = config.api.attention_slicing
        if slicing != "disabled" and is_pytorch_pipe(pipe) and not is_for_aitemplate:
            if slicing == "auto":
                pipe.enable_attention_slicing()
                logger.info("Optimization: Enabled attention slicing")
            else:
                pipe.enable_attention_slicing(slicing)
                logger.info(f"Optimization: Enabled attention slicing ({slicing})")

        # Change the order of the channels to be more efficient for the GPU
        # DirectML only supports contiguous memory format
        # Disable for IPEX as well, they don't like torch's way of setting memory format
        if (
            config.api.channels_last
            and config.api.device_type != "directml"
            and (not is_ipex_available() and config.api.device_type != "cpu")
            and not is_for_aitemplate
        ):
            pipe.unet.to(memory_format=torch.channels_last)  # type: ignore
            pipe.vae.to(memory_format=torch.channels_last)  # type: ignore
            logger.info("Optimization: Enabled channels_last memory format")

        # xFormers and SPDA
        if not is_for_aitemplate:
            set_attention_processor(pipe)

            if config.api.autocast:
                logger.info("Optimization: Enabled autocast")

        if can_offload:
            if not is_accelerate_available():
                logger.warning(
                    "Optimization: Offload is not available, because accelerate is not installed"
                )
            else:
                if offload == "model":
                    # Offload to CPU
                    from accelerate import cpu_offload_with_hook

                    if config.api.device_type == "cuda":
                        torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

                    hook = None

                    for cpu_offloaded_model in [
                        pipe.text_encoder,
                        pipe.unet,
                        pipe.vae,
                    ]:
                        _, hook = cpu_offload_with_hook(
                            cpu_offloaded_model, device, prev_module_hook=hook
                        )
                    pipe.final_offload_hook = hook
                    setattr(pipe.vae, "main_device", True)
                    setattr(pipe.unet, "main_device", True)
                    logger.info("Optimization: Offloaded model parts to CPU.")

                elif offload == "module":
                    # Enable sequential offload
                    from accelerate import cpu_offload, disk_offload

                    for m in [
                        pipe.vae,
                        pipe.unet,
                    ]:
                        if USE_DISK_OFFLOAD:
                            # If USE_DISK_OFFLOAD toggle set (idk why anyone would do this, but it's nice to support stuff
                            # like this in case anyone wants to try running this on fuck knows what)
                            # then offload to disk.
                            disk_offload(
                                m,
                                str(
                                    get_full_model_path(
                                        "offload-dir", model_folder="temp"
                                    )
                                    / m.__name__
                                ),
                                device,
                                offload_buffers=True,
                            )
                        else:
                            cpu_offload(m, device, offload_buffers=True)

                    logger.info("Optimization: Enabled sequential offload")

        if config.api.vae_slicing:
            if not (
                issubclass(pipe.__class__, StableDiffusionUpscalePipeline)
                or isinstance(pipe, StableDiffusionUpscalePipeline)
            ):
                pipe.enable_vae_slicing()
                logger.info("Optimization: Enabled VAE slicing")
            else:
                logger.debug(
                    "Optimization: VAE slicing is not available for upscale models"
                )
        if config.api.vae_tiling:
            if not (
                issubclass(pipe.__class__, StableDiffusionUpscalePipeline)
                or isinstance(pipe, StableDiffusionUpscalePipeline)
            ):
                pipe.enable_vae_tiling()
                logger.info("Optimization: Enabled VAE tiling")
            else:
                logger.debug(
                    "Optimization: VAE tiling is not available for upscale models"
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
        if config.api.device_type == "cpu":
            n = (cpu["num_virtual_cores"] // 4) * 3
            torch.set_num_threads(n)
            torch.set_num_interop_threads(n)

            logger.info(
                f"Running on an {cpu['VendorId']} device. Used threads: {torch.get_num_threads()}-{torch.get_num_interop_threads()} / {cpu['num_virtual_cores']}"
            )

            if is_ipex_available():
                import intel_extension_for_pytorch as ipex  # pylint: disable=import-error

                logger.info("Optimization: Running IPEX optimizations")

                if config.api.channels_last:
                    ipex.enable_auto_channels_last()
                else:
                    ipex.disable_auto_channels_last()
                ipex.enable_onednn_fusion(True)
                ipex.set_fp32_math_mode(
                    ipex.FP32MathMode.BF32
                    if "AMD" not in cpu["VendorId"]
                    else ipex.FP32MathMode.FP32
                )
                pipe.unet = ipex.optimize(
                    pipe.unet,  # type: ignore
                    dtype=config.api.dtype,
                    auto_kernel_selection=True,
                    sample_input=generate_inputs(config.api.dtype, device),
                )
                ipexed = True

        if config.api.device_type == "iree":
            convert_pipe_state_to_iree(pipe)  # type: ignore

        if config.api.trace_model and not ipexed and not is_for_aitemplate:
            logger.info("Optimization: Tracing model.")
            logger.warning("This will break controlnet and loras!")
            if config.api.attention_processor == "xformers":
                logger.warning(
                    "Skipping tracing because xformers used for attention processor. Please change to SDPA to enable tracing."
                )
            else:
                pipe.unet = trace_model(pipe.unet, config.api.dtype, device)  # type: ignore
        elif is_ipex_available() and config.api.trace_model and not is_for_aitemplate:
            logger.warning(
                "Skipping tracing because IPEX optimizations have already been done"
            )
            logger.warning(
                "This is a temporary measure, tracing will work with IPEX-enabled devices later on"
            )


def supports_tf32(device: Optional[torch.device] = None) -> bool:
    "Checks if device is post-Ampere"
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

    return issubclass(pipe.__class__, (DiffusionPipeline))
