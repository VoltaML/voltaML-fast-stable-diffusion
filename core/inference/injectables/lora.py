from typing import Dict, Union

import torch

from core.config import config

from .utils import HookObject


class LoRAModule(object):
    "Main module per LoRA object."

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.alpha: float = 1.0
        self.modules: Dict[str, torch.nn.Module] = {}


class LoRAUpDown(object):
    "A module containing LoRA 'displacer' modules."

    def __init__(self) -> None:
        self.down: Union[torch.nn.Conv2d, torch.nn.Linear] = None  # type: ignore
        self.up: Union[torch.nn.Conv2d, torch.nn.Linear] = None  # type: ignore
        self.alpha: float = 0.5


class LoRAManager(HookObject):
    "Manager for LoRA modules."

    def __init__(self) -> None:
        super().__init__(
            "lora",
            "lora",
            LoRAModule,
            [
                "Transformer2DModel",
                "Attention",
            ],
            [("lora_current_names", ()), ("lora_weigths_backup", None)],
        )

    @torch.no_grad()
    def load(
        self,
        name: str,
        state_dict: Dict[str, torch.nn.Module],
        modules: Dict[str, torch.nn.Module],
    ) -> LoRAModule:
        lora = LoRAModule(name)

        for k, v in state_dict.items():
            key, lora_key = k.split(".", 1)
            module = modules.get(key, None)
            if module is None:
                print(key, lora_key)
                continue
            lora_module = lora.modules.get(key, None)
            if lora_module is None:
                lora_module = LoRAUpDown()
                lora.modules[key] = lora_module  # type: ignore

            if lora_key == "alpha":
                lora_module.alpha = v.item()  # type: ignore
                continue
            if isinstance(module, torch.nn.Linear):
                module = torch.nn.Linear(v.shape[1], v.shape[0], bias=False)  # type: ignore
            else:
                module = torch.nn.Conv2d(v.shape[1], v.shape[0], (1, 1), bias=False)  # type: ignore

            with torch.no_grad():
                module.weight.copy_(v, True)  # type: ignore
            module.to(device=torch.device("cpu"), dtype=config.api.dtype)
            if lora_key == "lora_up.weight":
                lora_module.up = module
            else:
                lora_module.down = module
        print(*lora.modules.keys(), sep="\n")
        return lora

    def apply_hooks(self, p: Union[torch.nn.Conv2d, torch.nn.Linear]) -> None:
        current_names = getattr(p, "lora_current_names", ())
        wanted_names = tuple((x[0], x[1].alpha) for x in self.containers.items())

        weights_backup = getattr(p, "lora_weights_backup", None)
        if weights_backup is None:
            weights_backup = p.weight.to(torch.device("cpu"), copy=True)
            p.lora_weights_backup = weights_backup

        if current_names != wanted_names:
            if weights_backup is not None:
                p.weight.copy_(weights_backup)

            layer_name = getattr(p, "layer_name", None)
            for _, lora in self.containers.items():
                module: LoRAModule = lora.modules.get(layer_name, None)  # type: ignore
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
