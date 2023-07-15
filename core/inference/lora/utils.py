from pathlib import Path
from typing import Union, Dict, List, Optional

import torch

from ...config import config

torch.nn.Linear.old_forward = torch.nn.Linear.forward  # type: ignore
torch.nn.Conv2d.old_forward = torch.nn.Conv2d.forward  # type: ignore


class _LoRAUpDown(object):
    def __init__(self) -> None:
        self.down: Union[torch.nn.Conv2d, torch.nn.Linear] = None  # type: ignore
        self.up: Union[torch.nn.Conv2d, torch.nn.Linear] = None  # type: ignore
        self.alpha: float = 0.5


class _LoRAModule(object):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.alpha: float = 1.0
        self.modules: Dict[str, torch.nn.Module] = {}


class LoRAHookInjector(object):
    "LoRA hook manager class"

    def __init__(self):
        super().__init__()
        self.containers: Dict[str, _LoRAModule] = {}
        self.hooks = {}
        self.device: torch.device = None  # type: ignore
        self.dtype: torch.dtype = None  # type: ignore

    def _get_target_modules(
        self,
        root_module: torch.nn.Module,
        prefix: str,
        target_replace_modules: List[str],
    ):
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

    def change_forwards(self):
        "Redirect lora forward to this hook manager"
        d = self

        def lora_forward(self, input):  # pylint: disable=redefined-builtin
            d._apply_lora_state_dicts(self)  # pylint: disable=protected-access

            return self.old_forward(input)

        torch.nn.Linear.forward = lora_forward
        torch.nn.Conv2d.forward = lora_forward

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe"""
        assert len(self.hooks) == 0
        text_encoder_targets = self._get_target_modules(
            pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"]
        )
        unet_targets = self._get_target_modules(
            pipe.unet, "lora_unet", ["Transformer2DModel", "Attention"]
        )
        for name, target_module in text_encoder_targets + unet_targets:
            setattr(target_module, "lora_current_names", ())
            setattr(target_module, "lora_weights_backup", None)
            setattr(target_module, "lora_layer_name", name)
            self.hooks[name] = target_module

        self.change_forwards()

        self.device = config.api.device  # type: ignore
        self.dtype = pipe.unet.dtype

    def _apply_lora_state_dicts(self, p: Union[torch.nn.Conv2d, torch.nn.Linear]):
        current_names = getattr(p, "lora_current_names", ())
        wanted_names = tuple((x[0], x[1].alpha) for x in self.containers.items())

        weights_backup = getattr(p, "lora_weights_backup", None)
        if weights_backup is None:
            weights_backup = p.weight.to(torch.device("cpu"), copy=True)
            p.lora_weights_backup = weights_backup

        if current_names != wanted_names:
            if weights_backup is not None:
                p.weight.copy_(weights_backup)

            lora_layer_name = getattr(p, "lora_layer_name", None)
            for _, lora in self.containers.items():
                module: _LoRAModule = lora.modules.get(lora_layer_name, None)  # type: ignore
                if module is None:
                    continue
                with torch.no_grad():
                    up = module.up.weight.to(p.weight.device, dtype=p.weight.dtype)  # type: ignore
                    down = module.down.weight.to(p.weight.device, dtype=p.weight.dtype)  # type: ignore

                    if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                        updown = (
                            (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                            .unsqueeze(2)
                            .unsqueeze(3)
                        )
                    else:
                        updown = up @ down
                    p.weight += (
                        updown
                        * lora.alpha
                        * (
                            module.alpha / module.up.weight.shape[1]  # type: ignore
                            if module.alpha
                            else 1.0
                        )
                    )
            setattr(p, "lora_current_names", wanted_names)

    def uninstall_hooks(self):
        """Uninstall LoRAHook from the pipe."""
        self.hooks = {}

    def _load_lora(self, name: str, state_dict: Dict[str, torch.Tensor]) -> _LoRAModule:
        lora = _LoRAModule(name)
        for k, v in state_dict.items():
            key, lora_key = k.split(".", 1)
            module = self.hooks.get(key, None)
            if module is None:
                continue
            lora_module = lora.modules.get(key, None)
            if lora_module is None:
                lora_module = _LoRAUpDown()
                lora.modules[key] = lora_module  # type: ignore

            if lora_key == "alpha":
                lora_module.alpha = v.item()  # type: ignore
                continue
            if isinstance(module, torch.nn.Linear):
                module = torch.nn.Linear(v.shape[1], v.shape[0], bias=False)
            else:
                module = torch.nn.Conv2d(v.shape[1], v.shape[0], (1, 1), bias=False)

            with torch.no_grad():
                module.weight.copy_(v)
            module.to(device=torch.device("cpu"), dtype=config.api.dtype)
            if lora_key == "lora_up.weight":
                lora_module.up = module
            else:
                lora_module.down = module
        return lora

    def _load_state_dict(self, file: Path) -> Dict[str, torch.Tensor]:
        if file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(file)
        else:
            state_dict = torch.load(file)  # .bin, .pt, .ckpt...
        return state_dict

    def apply_lora(
        self, file: Union[Path, str], alpha: Optional[Union[torch.Tensor, float]] = None  # type: ignore
    ):
        """Load LoRA weights and apply LoRA to the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        if file.name in self.containers:
            return

        file: Path
        lora = self._load_lora(file.name, self._load_state_dict(file))
        lora.alpha = alpha if alpha else 1.0  # type: ignore
        self.containers[file.name] = lora

    def remove_lora(self, file: Union[Path, str]):
        """Remove the individual LoRA from the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        del self.containers[file.name]
