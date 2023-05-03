from typing import Tuple

from diffusers.models.unet_2d_condition import UNet2DConditionOutput

import torch


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
        self.in_channels = og.in_channels
        self.device = og.device if og.device else og.dev
        self.dtype = og.dtype
        self.config = og.config

    def forward(
        self, latent_model_input, t, encoder_hidden_states
    ) -> UNet2DConditionOutput:
        "Forward pass of the model"

        sample = self.og.forward(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)
