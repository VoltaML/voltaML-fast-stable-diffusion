from diffusers import UNet2DConditionModel
import torch

from core.config import config
from core.flags import LatentScaleModel
from .latents import scale_latents

step_limit = 0


class Scaler(torch.nn.Module):
    def __init__(
        self, scale: float, block: torch.nn.Module, scaler: LatentScaleModel
    ) -> None:
        super().__init__()
        self.scale = scale
        self.block = block
        self.scaler: LatentScaleModel = scaler

    def forward(self, hidden_states, *args, **kwargs):
        print(hidden_states.shape)
        # hidden_states = scale_latents(
        #    hidden_states, self.scale, self.scaler, config.api.deepshrink_antialias
        # )
        print(hidden_states.shape)
        if "scale" in kwargs:
            kwargs.pop("scale")
        return self.block(hidden_states, *args, **kwargs)


def modify_unet(
    unet: UNet2DConditionModel, step: int, total_steps: int
) -> UNet2DConditionModel:
    if not config.api.deepshrink_enabled:
        return unet

    global step_limit

    s1, s2 = config.api.deepshrink_stop_at_1, config.api.deepshrink_stop_at_2
    if s1 > s2:
        s2 = s1
    p1 = (s1, config.api.deepshrink_depth_1 - 1)
    p2 = (s2, config.api.deepshrink_depth_2 - 1)

    if step < step_limit:
        return unet

    for s, d in [p1, p2]:
        out_d = d if config.api.deepshrink_early_out else -(d + 1)
        if step < total_steps * s:
            if not isinstance(unet.down_blocks[d], Scaler):
                unet.down_blocks[d] = Scaler(
                    config.api.deepshrink_downscale,
                    unet.down_blocks[d],
                    config.api.deepshrink_scaler,
                )
                unet.up_blocks[out_d] = Scaler(
                    config.api.deepshrink_upscale,
                    unet.up_blocks[d],
                    config.api.deepshrink_scaler,
                )
            elif config.api.deepshrink_smooth_scaling:
                scale_ratio = step / (total_steps * s)
                downscale = min(
                    (1 - config.api.deepshrink_downscale) * scale_ratio
                    + config.api.deepshrink_downscale,
                    1.0,
                )
                unet.down_blocks[d].scale = downscale  # type: ignore
                unet.up_blocks[out_d].scale = config.api.deepshrink_upscale * (config.api.deepshrink_downscale / downscale)  # type: ignore
            return unet
        elif isinstance(unet.down_blocks[d], Scaler) and (p1[1] != p2[1] or s == p2[0]):
            unet.down_blocks[d] = unet.down_blocks[d].block
            unet.up_blocks[out_d] = unet.up_blocks[out_d].block
    step_limit = step
    return unet


def post_process(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    if not config.api.deepshrink_enabled:
        return unet

    for i, b in enumerate(unet.down_blocks):
        if isinstance(b, Scaler):
            unet.down_blocks[i] = b.block
    for i, b in enumerate(unet.up_blocks):
        if isinstance(b, Scaler):
            unet.up_blocks[i] = b.block

    global step_limit

    step_limit = 0
    return unet
