import logging
import warnings
from typing import Tuple, Union

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import is_accelerate_available, is_xformers_available
from packaging import version
from tqdm.auto import tqdm

from core.config import config
from core.files import get_full_model_path

logger = logging.getLogger(__name__)

USE_DISK_OFFLOAD = False

gpu_module = None
_device = None


def optimize_model(
    pipe: Union[StableDiffusionPipeline, StableDiffusionUpscalePipeline],
    device,
    use_fp32: bool,
    is_for_aitemplate: bool = False,
) -> None:
    "Optimize the model for inference"
    global _device  # pylint: disable=global-statement

    dtype = (
        torch.float32
        if use_fp32
        else (torch.bfloat16 if config.api.device_type == "cpu" else torch.float16)
    )
    pipe.to(device, torch_dtype=dtype)
    _device = device
    logger.info("Optimizing model")

    if config.api.device_type == "cuda":
        if config.api.reduced_precision:
            logger.info("Enabled reduced precision operations")
        torch.backends.cuda.matmul.allow_tf32 = config.api.reduced_precision  # type: ignore
        torch.backends.cudnn.allow_tf32 = config.api.reduced_precision  # type: ignore
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.api.reduced_precision  # type: ignore
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.api.reduced_precision  # type: ignore

        logger.info(
            f"CUDNN {'' if config.api.deterministic_generation else 'not '}using deterministic functions"
        )
        torch.backends.cudnn.deterministic = config.api.deterministic_generation  # type: ignore

        if config.api.cudnn_benchmark:
            logger.info("CUDNN wizardry enabled")
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
    if (
        config.api.channels_last
        and config.api.device_type != "directml"
        and not is_for_aitemplate
    ):
        pipe.unet.to(memory_format=torch.channels_last)  # type: ignore
        pipe.vae.to(memory_format=torch.channels_last)  # type: ignore
        logger.info("Optimization: Enabled channels_last memory format")

    # xFormers and SPDA
    if not is_for_aitemplate:
        if is_xformers_available() and config.api.attention_processor == "xformers":
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Optimization: Enabled xFormers memory efficient attention")
        elif version.parse(torch.__version__) >= version.parse("2.0.0"):
            from diffusers.models.attention_processor import AttnProcessor2_0

            pipe.unet.set_attn_processor(AttnProcessor2_0())  # type: ignore
            logger.info("Optimization: Enabled SDPA, because xformers is not installed")
        else:
            # This should only be the case if pytorch_directml is to be used
            # This isn't a hot-spot either, so it's fine (imo) to put in safety nets.
            from diffusers.models.attention_processor import AttnProcessor

            pipe.unet.set_attn_processor(AttnProcessor())  # type: ignore
            logger.info(
                "Optimization: Pytorch STILL not newer than 2.0.0, using Cross-Attention"
            )

    offload = (
        config.api.offload
        if (is_pytorch_pipe(pipe) and not is_for_aitemplate)
        else None
    )
    if config.api.device_type != "directml":
        if offload == "model":
            # Offload to CPU

            pipe.vae.to("cpu")  # type: ignore
            pipe.unet.to("cpu")  # type: ignore
            pipe.unet.register_forward_pre_hook(send_to_gpu)  # type: ignore
            pipe.vae.register_forward_pre_hook(send_to_gpu)  # type: ignore
            setattr(pipe.vae, "main_device", True)  # type: ignore
            setattr(pipe.unet, "main_device", True)  # type: ignore
            logger.info("Optimization: Offloaded VAE & UNet to CPU.")

        elif offload == "module":
            # Enable sequential offload

            if is_accelerate_available():
                from accelerate import cpu_offload, disk_offload

                for m in [
                    pipe.vae,  # type: ignore
                    pipe.safety_checker,  # type: ignore
                    pipe.unet,  # type: ignore
                ]:
                    if m is not None:
                        if USE_DISK_OFFLOAD:
                            # If LOW_RAM toggle set (idk why anyone would do this, but it's nice to support stuff
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
            else:
                logger.warning(
                    "Optimization: Sequential offload is not available, because accelerate is not installed"
                )

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

    if config.api.use_tomesd and not is_for_aitemplate:
        try:
            import tomesd

            tomesd.apply_patch(pipe.unet, ratio=config.api.tomesd_ratio, max_downsample=config.api.tomesd_downsample_layers)  # type: ignore
            logger.info("Optimization: Patched UNet for ToMeSD")
        except ImportError:
            logger.info(
                "Optimization: ToMeSD patch failed, despite having it enabled. Please check installation"
            )

    if config.api.trace_model:
        logger.info("Tracing model.")
        logger.warning("This will break controlnet and loras!")
        if config.api.attention_processor == "xformers":
            logger.warning(
                "Skipping tracing because xformers used for attention processor. Please change to SDPA to enable tracing."
            )
        else:
            pipe.unet = trace_model(pipe.unet, dtype, device)  # type: ignore

    logger.info("Optimization complete")


def generate_inputs(
    dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "Generate sample inputs for a conditional UNet2D"
    sample = torch.randn(2, 4, 64, 64).to(device, dtype=dtype)
    timestep = torch.rand(1).to(device, dtype=dtype) * 999
    encoder_hidden_states = torch.randn(2, 77, 768).to(device, dtype=dtype)
    return sample, timestep, encoder_hidden_states


def send_everything_to_cpu() -> None:
    "Offload module to CPU to save VRAM"

    global gpu_module  # pylint: disable=global-statement

    if gpu_module is not None:
        gpu_module.to("cpu")
    gpu_module = None


def send_to_gpu(module, _) -> None:
    "Load module back to GPU"

    global gpu_module  # pylint: disable=global-statement
    if gpu_module == module:
        return
    if gpu_module is not None:
        gpu_module.to("cpu")
    module.to(_device)
    gpu_module = module


def warmup(
    model: torch.nn.Module, amount: int, dtype: torch.dtype, device: torch.device
) -> None:
    "Warms up model with amount generated sample inputs."

    model.eval()
    with torch.inference_mode():
        for _ in tqdm(range(amount), unit="it", desc="Warming up", unit_scale=False):
            model(*generate_inputs(dtype, device))


def trace_model(
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    iterations: int = 25,
) -> torch.nn.Module:
    "Traces the model for inference"

    og = model
    from functools import partial

    model.forward = partial(model.forward, return_dict=False)
    warmup(model, iterations, dtype, device)
    if config.api.channels_last:
        model.to(memory_format=torch.channels_last)  # type: ignore
    logger.debug("Starting trace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if config.api.device_type == "cpu":
            torch.jit.enable_onednn_fusion(True)
        model = torch.jit.trace(model, generate_inputs(dtype, device), check_trace=False)  # type: ignore
        if config.api.device_type == "cpu":
            model = torch.jit.optimize_for_inference(model)  # type: ignore
    logger.debug("Tracing done")
    warmup(model, iterations // 5, dtype, device)

    class TracedUNet(torch.nn.Module):
        "UNet that was JIT traced and should be faster than the original"

        def __init__(self):
            super().__init__()
            self.in_channels = og.in_channels
            self.device = og.device
            self.dtype = og.dtype
            self.config = og.config

        def forward(
            self, latent_model_input, t, encoder_hidden_states
        ) -> UNet2DConditionOutput:
            "Forward pass of the model"

            sample = model(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    rn = TracedUNet()
    del og
    return rn


def is_pytorch_pipe(pipe):
    "Checks if the pipe is a pytorch pipe"

    return issubclass(pipe.__class__, (DiffusionPipeline))
