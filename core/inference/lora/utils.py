from pathlib import Path
from typing import Any, Union, Dict, List
import math

import torch

from ...config import config

class LoRAModule(torch.nn.Module):
    def __init__(
        self, org_module: torch.nn.Module, lora_dim: int = 4,
        alpha: float = 1.0, multiplier: Union[float, torch.Tensor] = 1.0
    ):
        super().__init__()

        module_name = org_module.__class__.__name__
        if module_name == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            kernel_size = stride = padding = None

        self.lora_dim = lora_dim
        self.lora_down = self._create_down_layer(module_name, in_dim, kernel_size, stride, padding)
        self.lora_up = self._create_up_layer(module_name, out_dim)

        self.register_buffer("alpha", torch.tensor(self._get_alpha(alpha)))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight)
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier

    def _create_down_layer(self, module_name: str, in_dim: int, kernel_size, stride, padding):
        if module_name == "Conv2d":
            return torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
        else:
            return torch.nn.Linear(in_dim, self.lora_dim, bias=False)

    def _create_up_layer(self, module_name: str, out_dim: int):
        if module_name == "Conv2d":
            return torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            return torch.nn.Linear(self.lora_dim, out_dim, bias=False)

    def _get_alpha(self, alpha: Union[float, torch.Tensor]):
        if alpha is None or alpha == 0:
            return self.lora_dim
        else:
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.detach().float().numpy()
            return alpha

    def forward(self, x: torch.Tensor):
        scale = self.alpha / self.lora_dim
        return self.multiplier * scale * self.lora_up(self.lora_down(x))

class LoRAModuleContainer(torch.nn.Module):
    "A container for lora hooks"
    def __init__(self, hooks: Dict[str, torch.nn.Module], state_dict: Dict[str, Any], multiplier: Union[float, torch.Tensor]):
        super().__init__()
        self.multiplier = multiplier

        # Create LoRAModule from state_dict information
        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_dim = value.size()[0]
                lora_name_alpha = key.split(".")[0] + '.alpha'
                alpha: Any = None
                if lora_name_alpha in state_dict:
                    alpha = state_dict[lora_name_alpha].item()
                hook = hooks[lora_name]
                lora_module = LoRAModule(
                    hook.orig_module, lora_dim=lora_dim, alpha=alpha, multiplier=multiplier
                )
                self.register_module(lora_name, lora_module)

        # Load whole LoRA weights
        self.load_state_dict(state_dict)

        # Register LoRAModule to LoRAHook
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                hook = hooks[name]
                hook.append_lora(module)
    @property
    def alpha(self):
        "The alpha (multiplier) of this lora container"
        return self.multiplier

    @alpha.setter
    def alpha(self, multiplier: float):
        self.multiplier = multiplier
        for _, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                module.multiplier = multiplier

    def _remove_from_hooks(self, hooks: Dict[str, LoRAModule]):
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                hook = hooks[name]
                hook.remove_lora(module)
                del module


class LoRAHook(torch.nn.Module):
    """
    replaces forward method of the original Linear,
    instead of replacing the original Linear module.
    """

    def __init__(self):
        super().__init__()
        self.lora_modules = []

    def install(self, orig_module: torch.nn.Module):
        assert not hasattr(self, "orig_module")
        self.orig_module = orig_module
        self.orig_forward = self.orig_module.forward
        self.orig_module.forward = self.forward

    def uninstall(self):
        assert hasattr(self, "orig_module")
        self.orig_module.forward = self.orig_forward
        del self.orig_forward
        del self.orig_module

    def append_lora(self, lora_module: LoRAModule):
        self.lora_modules.append(lora_module)

    def remove_lora(self, lora_module: LoRAModule):
        self.lora_modules.remove(lora_module)

    def forward(self, x: torch.Tensor):
        if len(self.lora_modules) == 0:
            return self.orig_forward(x)
        lora = torch.sum(torch.stack([lora(x) for lora in self.lora_modules]), dim=0)
        return self.orig_forward(x) + lora


class LoRAHookInjector(object):
    def __init__(self):
        super().__init__()
        self.containers: Dict[str, LoRAModuleContainer] = {}
        self.hooks = {}
        self.device: torch.device = None  # type: ignore
        self.dtype: torch.dtype = None  # type: ignore

    def _get_target_modules(self, root_module: torch.nn.Module, prefix: str, target_replace_modules: List[str]):
        target_modules = []
        for name, module in root_module.named_modules():
            if (
                module.__class__.__name__ in target_replace_modules
                and not "transformer_blocks" in name
            ):  # to adapt latest diffusers:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    if is_linear or is_conv2d:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        target_modules.append((lora_name, child_module))
        return target_modules

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe."""
        assert len(self.hooks) == 0
        text_encoder_targets = self._get_target_modules(
            pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"]
        )
        unet_targets = self._get_target_modules(
            pipe.unet, "lora_unet", ["Transformer2DModel", "Attention"]
        )
        for name, target_module in text_encoder_targets + unet_targets:
            hook = LoRAHook()
            hook.install(target_module)
            self.hooks[name] = hook

        self.device = config.api.device  # type: ignore
        self.dtype = pipe.unet.dtype

    def uninstall_hooks(self):
        """Uninstall LoRAHook from the pipe."""
        for _, v in self.hooks.items():
            v.uninstall()
        self.hooks = {}

    def apply_lora(self, file: Union[Path, str], alpha: Union[float, torch.Tensor] = 1.0):
        """Load LoRA weights and apply LoRA to the pipe."""
        assert len(self.hooks) != 0
        if not isinstance(file, Path):
            file = Path(file)

        # Unload if loaded already
        if file.name in self.containers:
            self.containers[file.name]._remove_from_hooks(self.hooks)  # pylint: disable=protected-access

        if file.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(file)
        else:
            state_dict = torch.load(file)  # .bin, etc...
        container = LoRAModuleContainer(self.hooks, state_dict, alpha)
        container.to(self.device, self.dtype)
        self.containers[file.name] = container

    def remove_lora(self, file: Union[Path, str]):
        """Remove the individual LoRA from the pipe."""
        if isinstance(file, str):
            file = Path(file)
        self.containers[file.name]._remove_from_hooks(self.hooks)  # pylint: disable=protected-access
        del self.containers[file.name]

    def _cleanup(self):
        if len(self.containers.keys()) == 0:
            self.uninstall_hooks()
