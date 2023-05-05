from typing import Tuple
import logging
import warnings

from diffusers.models.unet_2d_condition import UNet2DConditionOutput

import torch
from tqdm.auto import tqdm

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
        for _ in tqdm(range(amount), unit="it", desc="Warming up", unit_scale=False):
            model(*generate_inputs(dtype, device))


def trace_model(
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    iterations: int = 25,
    ipex: bool = False,
) -> torch.nn.Module:
    "Traces the model for inference"

    og = model
    from functools import partial

    if model.forward.__code__.co_argcount > 3:
        model.forward = partial(model.forward, return_dict=False)
    warmup(model, iterations, dtype, device)
    if config.api.channels_last and not ipex:
        model.to(memory_format=torch.channels_last)  # type: ignore
    logger.debug("Starting trace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if config.api.device_type == "cpu":
            torch.jit.enable_onednn_fusion(True)
        model = torch.jit.trace(model, generate_inputs(dtype, device), check_trace=False)  # type: ignore
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
