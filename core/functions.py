import logging
from pathlib import Path
from typing import Dict

from diffusers.utils import is_accelerate_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from PIL import Image

import torch

from core.files import get_full_model_path

logger = logging.getLogger(__name__)

# On a scale of 1 to 5, how much should we focus on speed over vram usage? 1 is speedy, 5 is vram efficient.
OPTIMIZATION_LEVEL = 4
LOW_RAM = False

# Variables when OPTIMIZATION_LEVEL == 4
gpu_module = None
_device = None

def optimize_model(pipe: StableDiffusionPipeline, device, use_f32: bool, override_optimization: int = 0) -> None:
    "Optimize the model for inference"
    global OPTIMIZATION_LEVEL, gpu_module, _device
    oldol = OPTIMIZATION_LEVEL
    if override_optimization != 0:
        OPTIMIZATION_LEVEL = override_optimization
    pipe.to(
        device, torch_dtype=torch.float16 if not use_f32 else torch.float32
    )
    _device = device

    logger.info("Optimizing model")

    if OPTIMIZATION_LEVEL == 3:
        logger.info("Optimization: Selected optimization preset: balanced")
    elif OPTIMIZATION_LEVEL < 3:
        logger.info("Optimization: Selected optimization preset: speed")
    else:
        logger.info("Optimization: Selected optimization preset: vram-balanced")

    if OPTIMIZATION_LEVEL >= 4:
        pipe.enable_attention_slicing(1)
        logger.info("Optimization: Enabled attention slicing (max)")
    elif OPTIMIZATION_LEVEL == 3:
        pipe.enable_attention_slicing()
        logger.info("Optimization: Enabled attention slicing")

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    logger.info("Optimization: Enabled channels_last memory format")

    if OPTIMIZATION_LEVEL >= 3:
        pipe.enable_vae_slicing()
        logger.info("Optimization: Enabled VAE slicing")

    try:
        # Put this here, 'cause (on my RTX 3080) only xformers + trace was faster than SDPA.
        import xformers
        # Maybe add a 6th optimization level (0) to enable tracing, since tracing takes about ~15 seconds on my machine.
        if OPTIMIZATION_LEVEL == 1:
            pipe.unet = trace_model(pipe.unet)
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Optimization: Enabled xFormers memory efficient attention")
    except ModuleNotFoundError:
        # Can't trace here, since traced models aren't unet2dconditionals.
        pipe.unet.set_attn_processor(AttnProcessor2_0())  # type: ignore
        logger.info("Optimization: Enabled SDPA, because xformers is not installed")

    # Changed order cause maybe that was the cause of all the problems..?
    # Started working immediately after, hope it wasn't just cpython caches...

    if OPTIMIZATION_LEVEL == 5:
        if is_accelerate_available():
            from accelerate import disk_offload, cpu_offload

            # Fuck you text_encoder, no one likes you
            for m in [pipe.vae, pipe.safety_checker, pipe.unet]: #, pipe.text_encoder]:
                if m is not None:
                    if LOW_RAM:
                        # If LOW_RAM toggle set (idk why anyone would do this, but it's nice to support stuff
                        # like this in case anyone wants to try running this on fuck knows what)
                        # then offload to disk.
                        disk_offload(m, str(get_full_model_path("offload-dir", model_folder="temp")), device, offload_buffers=True)
                    else:
                        cpu_offload(m, device, offload_buffers=True)
        
            logger.info("Optimization: Enabled sequential offload")
        else:
            logger.warn("Optimization: Sequential offload is not available, because accelerate is not installed")
    elif OPTIMIZATION_LEVEL == 4:
        pipe.vae.to("cpu")
        pipe.unet.to("cpu")
        pipe.unet.register_forward_pre_hook(send_to_gpu)
        pipe.vae.register_forward_pre_hook(send_to_gpu)
        setattr(pipe.vae, "main_device", True)
        setattr(pipe.unet, "main_device", True)
        logger.info("Optimization: Offloaded VAE & UNet to CPU.")

    logger.info("Optimization complete")
    OPTIMIZATION_LEVEL = oldol


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
    model.to(memory_format=torch.channels_last)
    model = torch.jit.trace(model, generate_inputs(), check_trace=False)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(*generate_inputs())

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = og.in_channels
            self.device = og.device
            self.dtype = og.dtype
            self.config = og.config

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = model(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)
    rn = TracedUNet()
    del og
    return rn

def send_everything_to_cpu():
    global gpu_module

    if gpu_module is not None:
        gpu_module.to("cpu")
    gpu_module = None

def send_to_gpu(module, _):
    global gpu_module
    if gpu_module == module:
        return
    if gpu_module is not None:
        gpu_module.to("cpu")
    module.to(_device)
    gpu_module = module
