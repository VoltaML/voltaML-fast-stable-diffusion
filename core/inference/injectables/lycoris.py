import logging
from typing import Any, Dict, Sized

import torch

from core.config import config

from .utils import HookObject

logger = logging.getLogger(__name__)


class LycoModule(object):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.alpha: float = 1.0
        self.dyn_dim = None
        self.modules = {}


class FullModule(object):
    def __init__(self) -> None:
        self.weight: torch.Tensor = None  # type: ignore
        self.alpha: float = None  # type: ignore
        self.scale: float = None  # type: ignore
        self.dim: int = None  # type: ignore
        self.shape: Sized = None  # type: ignore


class LycoUpDownModule(object):
    def __init__(self) -> None:
        self.up_module: object = None
        self.mid_module: object = None
        self.down_module: object = None
        self.alpha: float = None  # type: ignore
        self.scale: float = None  # type: ignore
        self.dim: int = None  # type: ignore
        self.shape: Sized = None  # type: ignore
        self.bias: torch.Tensor = None  # type: ignore


class LycoHadaModule(object):
    def __init__(self) -> None:
        self.t1: torch.Tensor = None  # type: ignore
        self.w1a: torch.Tensor = None  # type: ignore
        self.w1b: torch.Tensor = None  # type: ignore
        self.t2: torch.Tensor = None  # type: ignore
        self.w2a: torch.Tensor = None  # type: ignore
        self.w2b: torch.Tensor = None  # type: ignore
        self.alpha: float = None  # type: ignore
        self.scale: float = None  # type: ignore
        self.dim: int = None  # type: ignore
        self.shape: Sized = None  # type: ignore
        self.bias: torch.Tensor = None  # type: ignore


class IA3Module(object):
    def __init__(self) -> None:
        self.w = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.on_input = None


class LycoKronModule(object):
    def __init__(self) -> None:
        self.w1 = None
        self.w1a = None
        self.w1b = None
        self.w2 = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self._alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None

    @property
    def alpha(self):
        if self.w1a is None and self.w2a is None:
            return None
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, x):
        self._alpha = x


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


def make_weight_cp(t, wa, wb):
    temp = torch.einsum("i j k l, j r -> i r k l", t, wb)
    return torch.einsum("i j k l, i r -> r j k l", temp, wa)


CON_KEY = {
    "lora_up.weight",
    "dyn_up",
    "lora_down.weight",
    "dyn_down",
    "lora_mid.weight",
}
HADA_KEY = {
    "hada_t1",
    "hada_w1_a",
    "hada_w1_b",
    "hada_t2",
    "hada_w2_a",
    "hada_w2_b",
}
IA3_KEY = {"weight", "on_input"}
KRON_KEY = {
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_t2",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
}


class LyCORISManager(HookObject):
    def __init__(self) -> None:
        super().__init__(
            "lycoris",
            "lyco",
            LycoModule,
            [
                "Transformer2DModel",
                "Attention",
                "ResnetBlock2D",
                "Downsample2D",
                "Upsample2D",
                "TimestepEmbedding",
            ],
            [
                ("lyco_current_names", ()),
                ("lyco_weights_backup", None),
                ("lora_prev_names", ()),
            ],
        )

    @torch.no_grad()
    def load(
        self,
        name: str,
        state_dict: Dict[str, torch.nn.Module],
        modules: Dict[str, torch.nn.Module],
    ) -> Any:
        lyco = LycoModule(name)
        didntfind = []
        for k, v in state_dict.items():
            key, lyco_key = k.split(".", 1)
            sd_module = modules.get(key, None)
            if sd_module is None:
                if key not in didntfind:
                    didntfind.append(key)
                continue
            lyco_module: LycoUpDownModule = lyco.modules.get(key, None)
            if lyco_module is None:
                lyco_module = LycoUpDownModule()
                lyco.modules[key] = lyco_module
            if lyco_key == "alpha":
                lyco_module.alpha = v.item()  # type: ignore
                continue
            if lyco_key == "scale":
                lyco_module.scale = v.item()  # type: ignore
                continue
            if lyco_key == "diff":
                v = v.to(device=torch.device("cpu"), dtype=config.api.load_dtype)
                v.requires_grad_(False)
                lyco_module = FullModule()  # type: ignore
                lyco_module.weight = v  # type: ignore
                lyco.modules[key] = lyco_module
                continue
            if "bias_" in lyco_key:
                if lyco_module.bias is None:
                    lyco_module.bias = [None, None, None]
                if "bias_indices" == lyco_key:
                    lyco_module.bias[0] = v  # type: ignore
                elif "bias_values" == lyco_key:
                    lyco_module.bias[1] = v  # type: ignore
                elif "bias_size" == lyco_key:
                    lyco_module.bias[2] = v  # type: ignore

                if all((i is not None) for i in lyco_module.bias):
                    logger.debug("build bias")
                    lyco_module.bias = torch.sparse_coo_tensor(
                        lyco_module.bias[0],  # type: ignore
                        lyco_module.bias[1],  # type: ignore
                        tuple(lyco_module.bias[2]),  # type: ignore
                    ).to(device=torch.device("cpu"), dtype=config.api.load_dtype)
                    lyco_module.bias.requires_grad_(False)
                continue
            if lyco_key in CON_KEY:
                if isinstance(
                    sd_module,
                    (
                        torch.nn.Linear,
                        torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                        torch.nn.MultiheadAttention,
                    ),
                ):
                    v = v.reshape(v.shape[0], -1)  # type: ignore
                    module = torch.nn.Linear(v.shape[1], v.shape[0], bias=False)
                elif isinstance(sd_module, torch.nn.Conv2d):
                    if lyco_key == "lora_down.weight" or lyco_key == "dyn_up":
                        if len(v.shape) == 2:  # type: ignore
                            v = v.reshape(v.shape[0], -1, 1, 1)  # type: ignore
                        if v.shape[2] != 1 or v.shape[3] != 1:  # type: ignore
                            module = torch.nn.Conv2d(
                                v.shape[1],  # type: ignore
                                v.shape[0],  # type: ignore
                                sd_module.kernel_size,  # type: ignore
                                sd_module.stride,  # type: ignore
                                sd_module.padding,  # type: ignore
                                bias=False,
                            )
                        else:
                            module = torch.nn.Conv2d(
                                v.shape[1], v.shape[0], (1, 1), bias=False  # type: ignore
                            )
                    elif lyco_key == "lora_mid.weight":
                        module = torch.nn.Conv2d(
                            v.shape[1],  # type: ignore
                            v.shape[0],  # type: ignore
                            sd_module.kernel_size,  # type: ignore
                            sd_module.stride,  # type: ignore
                            sd_module.padding,  # type: ignore
                            bias=False,
                        )
                    elif lyco_key == "lora_up.weight" or lyco_key == "dyn_down":
                        module = torch.nn.Conv2d(
                            v.shape[1], v.shape[0], (1, 1), bias=False  # type: ignore
                        )

                if hasattr(sd_module, "weight"):
                    lyco_module.shape = sd_module.weight.shape  # type: ignore
                with torch.no_grad():
                    if v.shape != module.weight.shape:  # type: ignore
                        v = v.reshape(module.weight.shape)  # type: ignore
                    module.weight.copy_(v)  # type: ignore

                module.to(device=torch.device("cpu"), dtype=config.api.load_dtype)  # type: ignore
                module.requires_grad_(False)  # type: ignore

                if lyco_key == "lora_up.weight" or lyco_key == "dyn_up":
                    lyco_module.up_module = module  # type: ignore
                elif lyco_key == "lora_mid.weight":
                    lyco_module.mid_module = module  # type: ignore
                elif lyco_key == "lora_down.weight" or lyco_key == "dyn_down":
                    lyco_module.down_module = module  # type: ignore
                    lyco_module.dim = v.shape[0]  # type: ignore
                else:
                    logger.debug(f"invalid key {lyco_key}")
            elif lyco_key in HADA_KEY:
                if not isinstance(lyco_module, LycoHadaModule):
                    alpha = lyco_module.alpha
                    bias = lyco_module.bias
                    lyco_module = LycoHadaModule()  # type: ignore
                    lyco_module.alpha = alpha
                    lyco_module.bias = bias
                    lyco.modules[key] = lyco_module
                if hasattr(sd_module, "weight"):
                    lyco_module.shape = sd_module.weight.shape  # type: ignore

                v = v.to(device=torch.device("cpu"), dtype=config.api.load_dtype)
                v.requires_grad_(False)

                if lyco_key == "hada_w1_a":
                    lyco_module.w1a = v  # type: ignore
                elif lyco_key == "hada_w1_b":
                    lyco_module.w1b = v  # type: ignore
                    lyco_module.dim = v.shape[0]  # type: ignore
                elif lyco_key == "hada_w2_a":
                    lyco_module.w2a = v  # type: ignore
                elif lyco_key == "hada_w2_b":
                    lyco_module.w2b = v  # type: ignore
                    lyco_module.dim = v.shape[0]  # type: ignore
                elif lyco_key == "hada_t1":
                    lyco_module.t1 = v  # type: ignore
                elif lyco_key == "hada_t2":
                    lyco_module.t2 = v  # type: ignore

            elif lyco_key in IA3_KEY:
                if type(lyco_module) != IA3Module:
                    lyco_module = IA3Module()  # type: ignore
                    lyco.modules[key] = lyco_module

                if lyco_key == "weight":
                    lyco_module.w = v.to(torch.device("cpu"), dtype=config.api.load_dtype)  # type: ignore
                elif lyco_key == "on_input":
                    lyco_module.on_input = v  # type: ignore
            elif lyco_key in KRON_KEY:
                if not isinstance(lyco_module, LycoKronModule):
                    alpha = lyco_module.alpha
                    bias = lyco_module.bias
                    lyco_module = LycoKronModule()  # type: ignore
                    lyco_module.alpha = alpha
                    lyco_module.bias = bias  # type: ignore
                    lyco.modules[key] = lyco_module
                if hasattr(sd_module, "weight"):
                    lyco_module.shape = sd_module.weight.shape  # type: ignore

                v = v.to(device=torch.device("cpu"), dtype=config.api.load_dtype)
                v.requires_grad_(False)

                if lyco_key == "lokr_w1":
                    lyco_module.w1 = v  # type: ignore
                elif lyco_key == "lokr_w1_a":
                    lyco_module.w1a = v  # type: ignore
                elif lyco_key == "lokr_w1_b":
                    lyco_module.w1b = v  # type: ignore
                    lyco_module.dim = v.shape[0]  # type: ignore
                elif lyco_key == "lokr_w2":
                    lyco_module.w2 = v  # type: ignore
                elif lyco_key == "lokr_w2_a":
                    lyco_module.w2a = v  # type: ignore
                elif lyco_key == "lokr_w2_b":
                    lyco_module.w2b = v  # type: ignore
                    lyco_module.dim = v.shape[0]  # type: ignore
                elif lyco_key == "lokr_t2":
                    lyco_module.t2 = v  # type: ignore
        didntfind = list(
            filter(lambda x: "lora_te_text_model_encoder_layers_" not in x, didntfind)
        )
        if len(didntfind) != 0:
            logger.warning(f"Couldn't find the following weights: {didntfind}")
            logger.warning("The LyCORIS model could be broken?")
        return lyco

    def apply_hooks(self, p: torch.nn.Module) -> None:
        lyco_layer_name: str = getattr(p, "layer_name", None)  # type: ignore
        if lyco_layer_name is None:
            return

        current_names = getattr(p, "lyco_current_names", ())
        lora_prev_names = getattr(p, "lora_prev_names", ())
        lora_names = getattr(p, "lora_current_names", ())
        wanted_names = tuple(
            (x.name, x.alpha, x.dyn_dim) for x in self.containers.values()
        )

        # We take lora_changed as base_weight changed
        # but functional lora will not affect the weight so take it as unchanged
        lora_changed = lora_prev_names != lora_names

        lyco_changed = current_names != wanted_names

        weights_backup = getattr(p, "lyco_weights_backup", None)

        if (len(self.containers) and weights_backup is None) or (
            weights_backup is not None and lora_changed
        ):
            # backup when:
            #  * apply lycos but haven't backed up any weights
            #  * have outdated backed up weights
            if isinstance(p, torch.nn.MultiheadAttention):
                weights_backup = (
                    p.in_proj_weight.to(torch.device("cpu"), copy=True),
                    p.out_proj.weight.to(torch.device("cpu"), copy=True),
                )
            else:
                weights_backup = p.weight.to(torch.device("cpu"), copy=True)  # type: ignore
            p.lyco_weights_backup = weights_backup  # type: ignore
        elif len(self.containers) == 0:
            # when we unload all the lycos and have no weights to backup
            # clean backup weights to save ram
            p.lyco_weights_backup = None  # type: ignore

        if lyco_changed or lora_changed:
            if weights_backup is not None:
                if isinstance(p, torch.nn.MultiheadAttention):
                    p.in_proj_weight.copy_(weights_backup[0])
                    p.out_proj.weight.copy_(weights_backup[1])
                else:
                    p.weight.copy_(weights_backup)  # type: ignore

            for lyco in self.containers.values():
                module = lyco.modules.get(lyco_layer_name, None)
                multiplier = lyco.alpha
                if module is not None and hasattr(p, "weight"):
                    updown = _lyco_calc_updown(lyco, module, p.weight, multiplier)
                    if len(p.weight.shape) == 4 and p.weight.shape[1] == 9:  # type: ignore
                        # inpainting model. zero pad updown to make channel[1]  4 to 9
                        updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
                    p.weight += updown
                    continue

                module_q = lyco.modules.get(lyco_layer_name + "_q_proj", None)
                module_k = lyco.modules.get(lyco_layer_name + "_k_proj", None)
                module_v = lyco.modules.get(lyco_layer_name + "_v_proj", None)
                module_out = lyco.modules.get(lyco_layer_name + "_out_proj", None)

                if (
                    isinstance(p, torch.nn.MultiheadAttention)
                    and module_q
                    and module_k
                    and module_v
                    and module_out
                ):
                    updown_q = _lyco_calc_updown(
                        lyco, module_q, p.in_proj_weight, multiplier
                    )
                    updown_k = _lyco_calc_updown(
                        lyco, module_k, p.in_proj_weight, multiplier
                    )
                    updown_v = _lyco_calc_updown(
                        lyco, module_v, p.in_proj_weight, multiplier
                    )
                    updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                    p.in_proj_weight += updown_qkv  # type: ignore
                    p.out_proj.weight += _lyco_calc_updown(
                        lyco, module_out, p.out_proj.weight, multiplier
                    )
                    continue

                if module is None:
                    logging.debug(
                        f"Module {lyco_layer_name} could not be found. The models weights have somehow become tangled?"
                    )
                    continue

            setattr(p, "lora_prev_names", lora_names)
            setattr(p, "lyco_current_names", wanted_names)


def _rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def _rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum("n m k l, i n, m j -> i j k l", mid, up, down)


def _rebuild_weight(module, orig_weight: torch.Tensor, dyn_dim: int = None) -> torch.Tensor:  # type: ignore
    output_shape: Sized
    if module.__class__.__name__ == "LycoUpDownModule":
        up = module.up_module.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = module.down_module.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [up.size(0), down.size(1)]
        if (mid := module.mid_module) is not None:
            # cp-decomposition
            mid = mid.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = _rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = _rebuild_conventional(up, down, output_shape, dyn_dim)

    elif module.__class__.__name__ == "LycoHadaModule":
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]

        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape)

        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape)

        updown = updown1 * updown2

    elif module.__class__.__name__ == "FullModule":
        output_shape = module.weight.shape
        updown = module.weight.to(orig_weight.device, dtype=orig_weight.dtype)

    elif module.__class__.__name__ == "IA3Module":
        output_shape = [module.w.size(0), orig_weight.size(1)]
        if module.on_input:
            output_shape.reverse()
        else:
            module.w = module.w.reshape(-1, 1)
        updown = orig_weight * module.w.to(orig_weight.device, dtype=orig_weight.dtype)

    elif module.__class__.__name__ == "LycoKronModule":
        if module.w1 is not None:
            w1 = module.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b

        if module.w2 is not None:
            w2 = module.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        elif module.t2 is None:
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = make_weight_cp(t2, w2a, w2b)

        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        updown = make_kron(output_shape, w1, w2)

    else:
        raise NotImplementedError(
            f"Unknown module type: {module.__class__.__name__}\n"
            "If the type is one of "
            "'LycoUpDownModule', 'LycoHadaModule', 'FullModule', 'IA3Module', 'LycoKronModule'"
            "You may have other lyco extension that conflict with locon extension."
        )

    if hasattr(module, "bias") and module.bias is not None:
        updown = updown.reshape(module.bias.shape)
        updown += module.bias.to(orig_weight.device, dtype=orig_weight.dtype)
        updown = updown.reshape(output_shape)

    if len(output_shape) == 4:
        updown = updown.reshape(output_shape)  # type: ignore

    if orig_weight.size().numel() == updown.size().numel():
        updown = updown.reshape(orig_weight.shape)
    return updown


def _lyco_calc_updown(lyco, module, target, multiplier):
    updown = _rebuild_weight(module, target, lyco.dyn_dim)
    if lyco.dyn_dim and module.dim:
        dim = min(lyco.dyn_dim, module.dim)
    elif lyco.dyn_dim:
        dim = lyco.dyn_dim
    elif module.dim:
        dim = module.dim
    else:
        dim = None

    scale = (
        module.scale
        if module.scale is not None
        else module.alpha / dim
        if dim is not None and module.alpha is not None
        else 1.0
    )
    updown = updown * multiplier * scale
    return updown
