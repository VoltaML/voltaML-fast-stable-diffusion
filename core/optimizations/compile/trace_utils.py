import logging
import warnings
from typing import Tuple

import torch
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from tqdm import tqdm

from core.config import config

logger = logging.getLogger(__name__)


def generate_inputs(
    dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "Generate sample inputs for a conditional UNet2D"
    sample = torch.randn(2, 4, 64, 64).to(device, dtype=dtype)
    timestep = torch.rand(1).to(device, dtype=dtype) * 999
    encoder_hidden_states = torch.randn(2, 77, 768).to(device, dtype=dtype)
    return sample, timestep, encoder_hidden_states


class TracedUNet(torch.nn.Module):
    "UNet that was JIT traced and should be faster than the original"

    def __init__(self, og):
        super().__init__()
        self.og = og
        self.in_channels = og.in_channels if og.in_channels else 4
        self.device = og.device if og.device else og.dev
        self.dtype = og.dtype if og.dtype else torch.float32
        self.config = og.config if og.config else {}

    def forward(
        self, latent_model_input, t, encoder_hidden_states
    ) -> UNet2DConditionOutput:
        "Forward pass of the model"

        sample = self.og.forward(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


def warmup(
    model: torch.nn.Module, amount: int, dtype: torch.dtype, device: torch.device
) -> None:
    "Warms up model with amount generated sample inputs."

    model.eval()
    with torch.inference_mode():
        for _ in tqdm(range(amount), desc="Warming up"):
            model(*generate_inputs(dtype, device))


def trace_ipex(
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    cpu: dict,
) -> Tuple[torch.nn.Module, bool]:
    from core.inference.functions import is_ipex_available

    if is_ipex_available():
        import intel_extension_for_pytorch as ipex

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
        model = ipex.optimize(
            model,  # type: ignore
            dtype=dtype,
            auto_kernel_selection=True,
            sample_input=generate_inputs(dtype, device),
            concat_linear=True,
            graph_mode=True,
        )
        return model, True
    else:
        return model, False


def trace_model(
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    iterations: int = 25,
) -> torch.nn.Module:
    "Traces the model for inference"

    og = model
    from functools import partial

    if model.forward.__code__.co_argcount > 3:
        model.forward = partial(model.forward, return_dict=False)
    warmup(model, iterations, dtype, device)
    if config.api.channels_last:
        model.to(memory_format=torch.channels_last)  # type: ignore
    logger.debug("Starting trace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "cpu" in config.api.device:
            torch.jit.enable_onednn_fusion(True)
        model = torch.jit.trace(
            model, generate_inputs(dtype, device), check_trace=False
        )  # type: ignore
        model = torch.jit.freeze(model)  # type: ignore
    logger.debug("Tracing done")
    warmup(model, iterations // 5, dtype, device)

    model.in_channels = og.in_channels
    model.dtype = og.dtype
    model.device = og.device
    model.config = og.config

    rn = TracedUNet(model)
    del og
    return rn
