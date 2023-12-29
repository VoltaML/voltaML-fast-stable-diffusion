from typing import TYPE_CHECKING
import logging

import torch

from core.inference.utilities.load import load_checkpoint
from ..models.resnet import InflatedConv3d

if TYPE_CHECKING:
    from ..models.unet import UNet3DConditionModel


logger = logging.getLogger(__name__)


def patch_conv3d(unet: "UNet3DConditionModel", pia_path: str) -> "UNet3DConditionModel":
    old_weight, old_bias = unet.conv_in.weight, unet.conv_in.bias
    new_conv = InflatedConv3d(
        9,
        old_weight.shape[0],
        kernel_size=unet.conv_in.kernel_size,  # type: ignore
        stride=unet.conv_in.stride,  # type: ignore
        padding=unet.conv_in.padding,  # type: ignore
        bias=True if old_bias is not None else False,
    )
    param = torch.zeros((320, 5, 3, 3), requires_grad=True)
    new_conv.weight = torch.nn.Parameter(torch.cat([old_weight, param], dim=1))
    if old_bias is not None:
        new_conv.bias = old_bias
    unet.conv_in = new_conv
    unet.config["in_channels"] = 9

    checkpoint = load_checkpoint(pia_path, pia_path.endswith("safetensors"))
    m, u = unet.load_state_dict(checkpoint, strict=False)
    logger.debug(f"Missing keys: {m}, unexpected: {u}")
    return unet
