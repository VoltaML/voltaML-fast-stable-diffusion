from typing import TYPE_CHECKING
from safetensors.torch import load_file
import torch

from ..models.resnet import InflatedConv3d

if TYPE_CHECKING:
    from ..models.unet import UNet3DConditionModel


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

    if pia_path.endswith("safetensors"):
        state_dict = load_file(pia_path)
    else:
        state_dict = torch.load(pia_path)
    unet.load_state_dict(state_dict, strict=False)
    return unet
