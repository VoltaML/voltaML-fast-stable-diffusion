from typing import Tuple, Optional
from functools import partial

from diffusers import UNet2DConditionModel  # type: ignore
from diffusers.models.unet_2d_blocks import CrossAttnUpBlock2D, UpBlock2D
import torch

from core.flags import DeepshrinkFlag
from .latents import scale_latents

step_limit = 0


def nf(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    *args,
    **kwargs,
) -> torch.FloatTensor:
    mode = "bilinear"
    if hasattr(self, "kohya_scaler"):
        mode = self.kohya_scaler
    if mode == "bislerp":
        mode = "bilinear"
    out = list(res_hidden_states_tuple)
    for i, o in enumerate(out):
        if o.shape[2] != hidden_states.shape[2]:
            out[i] = torch.nn.functional.interpolate(
                o,
                (
                    hidden_states.shape[2],
                    hidden_states.shape[3],
                ),
                mode=mode,
            )
    res_hidden_states_tuple = tuple(out)

    return self.nn_forward(
        *args,
        hidden_states=hidden_states,
        res_hidden_states_tuple=res_hidden_states_tuple,
        **kwargs,
    )


CrossAttnUpBlock2D.nn_forward = CrossAttnUpBlock2D.forward  # type: ignore
UpBlock2D.nn_forward = UpBlock2D.forward  # type: ignore
CrossAttnUpBlock2D.forward = nf
UpBlock2D.forward = nf


def modify_unet(
    unet: UNet2DConditionModel,
    step: int,
    total_steps: int,
    flag: Optional[DeepshrinkFlag] = None,
) -> UNet2DConditionModel:
    if flag is None:
        return unet

    global step_limit

    s1, s2 = flag.stop_at_1, flag.stop_at_2
    if s1 > s2:
        s2 = s1
    p1 = (s1, flag.depth_1 - 1)
    p2 = (s2, flag.depth_2 - 1)

    if step < step_limit:
        return unet

    for s, d in [p1, p2]:
        out_d = d if flag.early_out else -(d + 1)
        out_d = min(out_d, len(unet.up_blocks) - 1)
        if step < total_steps * s:
            if not hasattr(unet.down_blocks[d], "kohya_scale"):
                for block, scale in [
                    (unet.down_blocks[d], flag.base_scale),
                    (unet.up_blocks[out_d], 1.0 / flag.base_scale),
                ]:
                    setattr(block, "kohya_scale", scale)
                    setattr(block, "kohya_scaler", flag.scaler)
                    setattr(block, "_orignal_forawrd", block.forward)

                    def new_forawrd(self, hidden_states, *args, **kwargs):
                        hidden_states = scale_latents(
                            hidden_states,
                            self.kohya_scale,
                            self.kohya_scaler,
                            False,
                        )
                        if "scale" in kwargs:
                            kwargs.pop("scale")
                        return self._orignal_forawrd(hidden_states, *args, **kwargs)

                    block.forward = partial(new_forawrd, block)
            return unet
        elif hasattr(unet.down_blocks[d], "kohya_scale") and (
            p1[1] != p2[1] or s == p2[0]
        ):
            unet.down_blocks[d].forward = unet.down_blocks[d]._orignal_forawrd
            if hasattr(unet.up_blocks[out_d], "_orignal_forawrd"):
                unet.up_blocks[out_d].forward = unet.up_blocks[out_d]._orignal_forawrd
    step_limit = step
    return unet


def post_process(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    for i, b in enumerate(unet.down_blocks):
        if hasattr(b, "kohya_scale"):
            unet.down_blocks[i].forward = b._orignal_forawrd
    for i, b in enumerate(unet.up_blocks):
        if hasattr(b, "kohya_scale"):
            unet.up_blocks[i].forward = b._orignal_forawrd

    global step_limit

    step_limit = 0
    return unet
