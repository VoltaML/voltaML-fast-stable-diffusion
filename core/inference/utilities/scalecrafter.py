from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import yaml
import math
from pathlib import Path

from diffusers.models.unet_2d_condition import UNet2DConditionModel
import torch
import scipy


@dataclass
class ScalecrafterSettings:
    inflate_tau: float
    ndcfg_tau: float
    dilate_tau: float

    progressive: bool

    dilation_settings: Dict[str, float]
    ndcfg_dilate_settings: Dict[str, float]
    disperse_list: List[str]
    disperse: Optional[torch.Tensor]

    height: int = 0
    width: int = 0
    base: str = "sd15"


SCALECRAFTER_DIR = Path("data/scalecrafter")


_unet_inflate, _unet_inflate_vanilla = None, None
_backup_forwards = dict()


def find_config_closest_to(
    base: str, height: int, width: int, disperse: bool = False
) -> ScalecrafterSettings:
    """Find ScaleCrafter config for specified SDxx version closest to provided resolution."""
    # Normalize base to the format in SCALECRAFTER_CONFIG
    base = base.replace(".", "").lower()
    if base == "sd1x":
        base = "sd15"
    elif base == "sd2x":
        base = "sd21"

    resolutions = [
        x
        for x in SCALECRAFTER_CONFIG
        if x.base == base and ((x.disperse is not None) == disperse)
    ]

    # If there are no resolutions for said base, use default one.
    if len(resolutions) == 0:
        resolutions = [SCALECRAFTER_CONFIG[0]]

    # Map resolutions to a tuple of (name, resolution -> h*w)
    resolutions = [
        (x, abs((x.height * x.width * 64) - (height * width))) for x in resolutions
    ]

    # Read the settings of the one with the lowest resolution.
    return min(resolutions, key=lambda x: x[1])[0]


class ReDilateConvProcessor:
    "Conv2d with support for up-/downscaling latents"

    def __init__(
        self,
        module: torch.nn.Conv2d,
        pf_factor: float = 1.0,
        mode: str = "bilinear",
        activate: bool = True,
    ):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate

    def __call__(
        self, input: torch.Tensor, scale: float, *args, **kwargs
    ) -> torch.Tensor:
        if len(args) > 0:
            print(len(args))
            print("".join(map(str, map(type, args))))
        if self.activate:
            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            inflation_kernel_size = (self.module.weight.shape[-1] - 3) // 2
            self.module.dilation, self.module.padding = self.dilation, (  # type: ignore
                self.dilation * (1 + inflation_kernel_size),
                self.dilation * (1 + inflation_kernel_size),
            )
            ori_size, new_size = (
                (
                    int(input.shape[-2] / self.module.stride[0]),
                    int(input.shape[-1] / self.module.stride[1]),
                ),
                (
                    round(input.shape[-2] * self.factor),
                    round(input.shape[-1] * self.factor),
                ),
            )
            input = torch.nn.functional.interpolate(
                input, size=new_size, mode=self.mode
            )
            input = self.module._conv_forward(
                input, self.module.weight, self.module.bias
            )
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            result = torch.nn.functional.interpolate(
                input, size=ori_size, mode=self.mode
            )
            return result
        else:
            return self.module._conv_forward(
                input, self.module.weight, self.module.bias
            )


def inflate_kernels(
    unet: UNet2DConditionModel,
    inflate_conv_list: list,
    inflation_transform: torch.Tensor,
) -> UNet2DConditionModel:
    def replace_module(module: torch.nn.Module, name: List[str], index: list, value):
        if len(name) == 1 and len(index) == 0:
            setattr(module, name[0], value)
            return module

        current_name, next_name = name[0], name[1:]
        current_index, next_index = int(index[0]), index[1:]
        replace = getattr(module, current_name)
        replace[current_index] = replace_module(
            replace[current_index], next_name, next_index, value
        )
        setattr(module, current_name, replace)
        return module

    inflation_transform.to(dtype=unet.dtype, device=unet.device)

    for name, module in unet.named_modules():
        if name in inflate_conv_list:
            weight, bias = module.weight.detach(), module.bias.detach()
            (i, o, *_), kernel_size = (
                weight.shape,
                int(math.sqrt(inflation_transform.shape[0])),
            )
            transformed_weight = torch.einsum(
                "mn, ion -> iom",
                inflation_transform.to(dtype=weight.dtype),
                weight.view(i, o, -1),
            )
            conv = torch.nn.Conv2d(
                o,
                i,
                (kernel_size, kernel_size),
                stride=module.stride,
                padding=module.padding,
                device=weight.device,
                dtype=weight.dtype,
            )
            conv.weight.detach().copy_(
                transformed_weight.view(i, o, kernel_size, kernel_size)
            )
            conv.bias.detach().copy_(bias)  # type: ignore

            sub_names = name.split(".")
            if name.startswith("mid_block"):
                names, indexes = sub_names[1::2], sub_names[2::2]
                unet.mid_block = replace_module(unet.mid_block, names, indexes, conv)  # type: ignore
            else:
                names, indexes = sub_names[0::2], sub_names[1::2]
                replace_module(unet, names, indexes, conv)
    return unet


def scale_setup(unet: UNet2DConditionModel, settings: Optional[ScalecrafterSettings]):
    global _unet_inflate, _unet_inflate_vanilla

    if _unet_inflate_vanilla is not None:
        del _unet_inflate_vanilla, _unet_inflate

    if settings is None:
        return

    if settings.disperse is not None:
        if len(settings.disperse_list) != 0:
            _unet_inflate = deepcopy(unet)
            _unet_inflate = inflate_kernels(
                _unet_inflate, settings.disperse_list, settings.disperse
            )
            if settings.ndcfg_tau > 0:
                _unet_inflate_vanilla = deepcopy(unet)
                _unet_inflate_vanilla = inflate_kernels(
                    _unet_inflate_vanilla, settings.disperse_list, settings.disperse
                )


def scale(
    unet: UNet2DConditionModel,
    settings: Optional[ScalecrafterSettings],
    step: int,
    total_steps: int,
) -> UNet2DConditionModel:
    if settings is None:
        return unet

    global _backup_forwards, _unet_inflate, _unet_inflate_vanilla

    tau = step / total_steps
    inflate = settings.inflate_tau < tau and settings.disperse is not None

    if inflate:
        unet = _unet_inflate  # type: ignore

    for name, module in unet.named_modules():
        if settings.dilation_settings is not None:
            if name in settings.dilation_settings.keys():
                _backup_forwards[name] = module.forward
                dilate = settings.dilation_settings[name]
                if settings.progressive:
                    dilate = max(
                        math.ceil(
                            dilate * ((settings.dilate_tau - tau) / settings.dilate_tau)
                        ),
                        2,
                    )
                if tau < settings.inflate_tau and name in settings.disperse_list:
                    dilate = dilate / 2
                module.forward = ReDilateConvProcessor(  # type: ignore
                    module, dilate, mode="bilinear", activate=tau < settings.dilate_tau  # type: ignore
                )

    return unet


def post_scale(
    unet: UNet2DConditionModel,
    settings: Optional[ScalecrafterSettings],
    step: int,
    total_steps: int,
    call,
    *args,
    **kwargs,
) -> Tuple[UNet2DConditionModel, Optional[torch.Tensor]]:
    if settings is None:
        return unet, None

    global _backup_forwards
    for name, module in unet.named_modules():
        if name in _backup_forwards.keys():
            module.forward = _backup_forwards[name]
    _backup_forwards.clear()

    tau = step / total_steps
    noise_pred_vanilla = None
    if tau < settings.ndcfg_tau:
        inflate = settings.inflate_tau < tau and settings.disperse is not None

        if inflate:
            unet = _unet_inflate_vanilla  # type: ignore

        for name, module in unet.named_modules():
            if name in settings.ndcfg_dilate_settings.keys():
                _backup_forwards[name] = module.forward
                dilate = settings.ndcfg_dilate_settings[name]
                if settings.progressive:
                    dilate = max(
                        math.ceil(
                            dilate * ((settings.ndcfg_tau - tau) / settings.ndcfg_tau)
                        ),
                        2,
                    )
                if tau < settings.inflate_tau and name in settings.disperse_list:
                    dilate = dilate / 2
                module.forward = ReDilateConvProcessor(  # type: ignore
                    module, dilate, mode="bilinear", activate=tau < settings.ndcfg_tau  # type: ignore
                )
        noise_pred_vanilla = call(*args, **kwargs)

        for name, module in unet.named_modules():
            if name in _backup_forwards.keys():
                module.forward = _backup_forwards[name]
        _backup_forwards.clear()

    return unet, noise_pred_vanilla


class ScaledAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, processor, test_res, train_res):
        self.processor = processor
        self.test_res = test_res
        self.train_res = train_res

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        input_ndim = hidden_states.ndim
        if encoder_hidden_states is None:
            if input_ndim == 4:
                _, _, height, width = hidden_states.shape
                sequence_length = height * width
            else:
                _, sequence_length, _ = hidden_states.shape

            test_train_ratio = float(self.test_res / self.train_res)
            train_sequence_length = sequence_length / test_train_ratio
            scale_factor = math.log(sequence_length, train_sequence_length) ** 0.5
        else:
            scale_factor = 1

        original_scale = attn.scale
        attn.scale = attn.scale * scale_factor
        hidden_states = self.processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )
        attn.scale = original_scale
        return hidden_states


def read_settings(config_name: str):
    file = SCALECRAFTER_DIR / "configs" / config_name
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    # 0. Default height and width to unet
    base = config_name.split("_")[0].strip().lower().replace(".", "")
    steps = config["num_inference_steps"]
    height = config["latent_height"]
    width = config["latent_width"]
    inflate_tau = config["inflate_tau"] / steps
    ndcfg_tau = config["ndcfg_tau"] / steps
    dilate_tau = config["dilate_tau"] / steps
    progressive = config["progressive"]

    dilate_settings = dict()
    if config["dilate_settings"] is not None:
        with open(os.path.join(SCALECRAFTER_DIR, config["dilate_settings"])) as f:
            for line in f.readlines():
                name, dilate = line.strip().split(":")
                dilate_settings[name] = float(dilate)

    ndcfg_dilate_settings = dict()
    if config["ndcfg_dilate_settings"] is not None:
        with open(os.path.join(SCALECRAFTER_DIR, config["ndcfg_dilate_settings"])) as f:
            for line in f.readlines():
                name, dilate = line.strip().split(":")
                ndcfg_dilate_settings[name] = float(dilate)

    inflate_settings = list()
    if config["disperse_settings"] is not None:
        with open(os.path.join(SCALECRAFTER_DIR, config["disperse_settings"])) as f:
            inflate_settings = list(map(lambda x: x.strip(), f.readlines()))

    disperse = None
    if config["disperse_transform"] is not None:
        disperse = scipy.io.loadmat(
            os.path.join(SCALECRAFTER_DIR, config["disperse_transform"])
        )["R"]
        disperse = torch.tensor(disperse, device="cpu")

    return ScalecrafterSettings(
        inflate_tau,
        ndcfg_tau,
        dilate_tau,
        # --
        progressive,
        # --
        dilate_settings,
        ndcfg_dilate_settings,
        inflate_settings,
        # --
        disperse=disperse,
        height=height,
        width=width,
        base=base,
    )


SCALECRAFTER_CONFIG = list(map(read_settings, os.listdir(SCALECRAFTER_DIR / "configs")))
