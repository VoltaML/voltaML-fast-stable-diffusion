import logging
from pathlib import Path
from typing import Dict

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.utils import is_accelerate_available
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image

from core.config import config
from core.files import get_full_model_path

logger = logging.getLogger(__name__)

USE_DISK_OFFLOAD = False

# Variables when config.api.optLevel == 4
gpu_module = None
_device = None


def optimize_model(pipe: StableDiffusionPipeline, device, use_f32: bool) -> None:
    "Optimize the model for inference"
    global _device  # pylint: disable=global-statement

    pipe.to(device, torch_dtype=torch.float16 if not use_f32 else torch.float32)
    _device = device

    logger.info("Optimizing model")

    # Attention slicing that should save VRAM (but is slower)
    if config.api.optLevel >= 3:
        pipe.enable_attention_slicing(1)
        logger.info("Optimization: Enabled attention slicing (max)")
    elif config.api.optLevel == 2:
        pipe.enable_attention_slicing()
        logger.info("Optimization: Enabled attention slicing")

    # Change the order of the channels to be more efficient for the GPU
    pipe.unet.to(memory_format=torch.channels_last)  # type: ignore
    pipe.vae.to(memory_format=torch.channels_last)  # type: ignore
    logger.info("Optimization: Enabled channels_last memory format")

    # xFormers and SPDA
    if is_xformers_available():
        if config.api.optLevel == 0:
            logger.info("Optimization: Tracing model")
            pipe.unet = trace_model(pipe.unet)  # type: ignore
            logger.info("Optimization: Model successfully traced")
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Optimization: Enabled xFormers memory efficient attention")
    else:
        pipe.unet.set_attn_processor(AttnProcessor2_0())  # type: ignore
        logger.info("Optimization: Enabled SDPA, because xformers is not installed")

    if config.api.optLevel == 3:
        # Offload to CPU

        pipe.vae.to("cpu")  # type: ignore
        pipe.unet.to("cpu")  # type: ignore
        pipe.unet.register_forward_pre_hook(send_to_gpu)  # type: ignore
        pipe.vae.register_forward_pre_hook(send_to_gpu)  # type: ignore
        setattr(pipe.vae, "main_device", True)  # type: ignore
        setattr(pipe.unet, "main_device", True)  # type: ignore
        logger.info("Optimization: Offloaded VAE & UNet to CPU.")

    elif config.api.optLevel == 4:
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
                                get_full_model_path("offload-dir", model_folder="temp")
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

    pipe.enable_vae_slicing()
    logger.info("Optimization: Enabled VAE slicing")

    logger.info("Optimization complete")


def image_meta_from_file(path: Path) -> Dict[str, str]:
    "Return image metadata from a file"

    with path.open("rb") as f:
        image = Image.open(f)
        text = image.text  # type: ignore

        return text


def trace_model(model: torch.nn.Module) -> torch.nn.Module:
    "Traces the model for inference"

    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states

    og = model
    model.eval()
    from functools import partial

    model.forward = partial(model.forward, return_dict=False)
    with torch.inference_mode():
        for _ in range(5):
            model(*generate_inputs())
    model.to(memory_format=torch.channels_last)  # type: ignore
    model = torch.jit.trace(model, generate_inputs(), check_trace=False)  # type: ignore
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(*generate_inputs())

    class TracedUNet(torch.nn.Module):
        "UNet that was JIT traced and should be faster than the original"

        def __init__(self):
            super().__init__()
            self.in_channels = og.in_channels
            self.device = og.device
            self.dtype = og.dtype
            self.config = og.config

        def forward(self, latent_model_input, t, encoder_hidden_states):
            "Forward pass of the model"

            sample = model(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    rn = TracedUNet()
    del og
    return rn


def send_everything_to_cpu():
    "Offload module to CPU to save VRAM"

    global gpu_module  # pylint: disable=global-statement

    if gpu_module is not None:
        gpu_module.to("cpu")
    gpu_module = None


def send_to_gpu(module, _):
    "Load module back to GPU"

    global gpu_module  # pylint: disable=global-statement
    if gpu_module == module:
        return
    if gpu_module is not None:
        gpu_module.to("cpu")
    module.to(_device)
    gpu_module = module
