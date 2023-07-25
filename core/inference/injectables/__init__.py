from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from ...config import config
from .lora import LoRAManager
from .lycoris import LyCORISManager
from .utils import HookObject

torch.nn.Linear.old_forward = torch.nn.Linear.forward  # type: ignore
torch.nn.Conv2d.old_forward = torch.nn.Conv2d.forward  # type: ignore


def load_lora_utilities(pipe):
    "Reset/redirect Linear and Conv2ds forward to the lora processor"
    if hasattr(pipe, "lora_injector"):
        pipe.lora_injector.change_forwards()
    else:
        torch.nn.Linear.forward = torch.nn.Linear.old_forward  # type: ignore
        torch.nn.Conv2d.forward = torch.nn.Conv2d.old_forward  # type: ignore


def install_lora_hook(pipe):
    "Install LoRAHook to the pipe"
    if hasattr(pipe, "lora_injector"):
        return
    injector = HookManager()
    injector.install_hooks(pipe)
    pipe.lora_injector = injector


def uninstall_lora_hook(pipe):
    "Remove LoRAHook from the pipe"
    del pipe.lora_injector


class HookManager(object):
    "Hook manager class for everything that can be injected into a pipeline. This usually includes stuff like LoRA and LyCORIS"

    def __init__(self):
        super().__init__()
        self.managers: List[HookObject] = []
        self.modules = {}
        self.device: torch.device = None  # type: ignore
        self.dtype: torch.dtype = None  # type: ignore

        self._load_modules()

    def _load_modules(self):
        self.managers.append(LoRAManager())
        self.managers.append(LyCORISManager())

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

    def _load_state_dict(self, file: Union[Path, str]) -> Dict[str, torch.nn.Module]:
        if file is not Path:
            file = Path(file)
        if file.suffix == ".safetensors":  # type: ignore
            from safetensors.torch import load_file

            state_dict = load_file(file)
        else:
            state_dict = torch.load(file)  # .bin, .pt, .ckpt...
        return state_dict  # type: ignore

    @torch.no_grad()
    def apply_weights(self, p) -> None:
        "Apply the modifications of the hooks"
        for m in self.managers:
            m.apply_hooks(p)

    def change_forwards(self):
        "Redirect lora forward to this hook manager"
        d = self

        def lora_forward(self, input):  # pylint: disable=redefined-builtin
            d.apply_weights(self)

            return self.old_forward(input)

        torch.nn.Linear.forward = lora_forward
        torch.nn.Conv2d.forward = lora_forward

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe"""
        assert len(self.modules) == 0
        text_encoder_targets = self._get_target_modules(
            pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"]
        )
        targets = []
        for m in self.managers:
            for target in m.targets:
                if target not in targets:
                    targets.append(target)
        unet_targets = self._get_target_modules(
            pipe.unet,
            "lora_unet",
            targets,
        )
        for name, target_module in text_encoder_targets + unet_targets:
            for m in self.managers:
                for attr in m.default_attributes:
                    setattr(target_module, attr[0], attr[1])
            setattr(target_module, "layer_name", name)
            self.modules[name] = target_module

        self.change_forwards()

        self.device = config.api.device  # type: ignore
        self.dtype = pipe.unet.dtype

    # Temporary, TODO: replace this with something sensible
    def apply_lycoris(
        self, file: Union[Path, str], alpha: Optional[float] = None  # type: ignore
    ):
        """Load LyCORIS weights and apply it to the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        if file.name in self.managers[1].containers:
            return
        file: Path
        lyco = self.managers[1].load(
            file.name, self._load_state_dict(file), self.modules
        )
        lyco.alpha = alpha if alpha else 1.0
        self.managers[1].containers[file.name] = lyco

    def remove_lycoris(self, file: Union[Path, str]):
        """Remove the individual LyCORIS form the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        del self.managers[1].containers[file.name]

    def apply_lora(
        self, file: Union[Path, str], alpha: Optional[Union[torch.Tensor, float]] = None  # type: ignore
    ):
        """Load LoRA weights and apply LoRA to the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        if file.name in self.managers[0].containers:
            return
        file: Path
        lora = self.managers[0].load(
            file.name, self._load_state_dict(file), self.modules
        )
        lora.alpha = alpha if alpha else 1.0
        self.managers[0].containers[file.name] = lora

    def remove_lora(self, file: Union[Path, str]):
        """Remove the individual LoRA from the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        del self.managers[0].containers[file.name]
