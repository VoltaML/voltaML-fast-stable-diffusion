from typing import Dict, Union
import logging

import torch

from core.config import config

from .utils import HookObject

logger = logging.getLogger(__name__)


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
        self.alpha: float = 1.0  # why was it 0.5???


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

        missing = 0

        for k, v in state_dict.items():
            key, lora_key = k.split(".", 1)
            module = modules.get(key, None)

            if module is None:
                # Big problem! Something broke, and that's BADDDDD!!!
                logger.debug(f"Couldn't find {key}.{lora_key}")
                missing += 1
                continue
            lora_module = lora.modules.get(key, None)
            if lora_module is None:
                lora_module = LoRAUpDown()
                lora.modules[key] = lora_module  # type: ignore

            if lora_key == "alpha":
                lora_module.alpha = v  # type: ignore
                continue
            if isinstance(v, float):
                # Probably loaded wrong, or lora is broken: just ignore, it's gonna be fine...
                logger.debug(
                    f"{key}.{lora_key} has for whatever reason a float here, when it shouldn't..."
                )
                continue
            if isinstance(module, torch.nn.Linear):
                module = torch.nn.Linear(v.shape[1], v.shape[0], bias=False)  # type: ignore
            else:
                module = torch.nn.Conv2d(
                    v.shape[1], v.shape[0], v.shape[2], v.shape[3], bias=False
                )  # type: ignore

            with torch.no_grad():
                module.weight.copy_(v, True)  # type: ignore
            module.to(device=torch.device("cpu"), dtype=config.api.load_dtype)
            if lora_key == "lora_up.weight":
                lora_module.up = module
            else:
                lora_module.down = module
        if missing != 0:
            logger.error(
                f"Uh oh! Something went wrong loading the lora. If the output looks completely whack, contact us on discord! Missing keys: {missing}."
            )
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
            skipped = False
            for _, lora in self.containers.items():
                module: LoRAModule = lora.modules.get(layer_name, None)  # type: ignore
                if module is None:
                    continue
                if module.up is None:  # type: ignore
                    skipped = True
                    continue
                with torch.no_grad():
                    weight = p.weight.clone()
                    if hasattr(p, "fp16_weight"):
                        weight = p.fp16_weight.clone()  # type: ignore
                    up = module.up.weight.to(p.weight.device, dtype=weight.dtype)  # type: ignore
                    down = module.down.weight.to(p.weight.device, dtype=weight.dtype)  # type: ignore

                    if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                        updown = (
                            (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                            .unsqueeze(2)
                            .unsqueeze(3)
                        )
                    else:
                        updown = up @ down

                    if len(weight.shape) == 4 and weight.shape[1] == 9:
                        # inpainting model. zero pad updown to make channel[1] 4 -> 9
                        updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
                    weight += (
                        updown
                        * lora.alpha
                        * (
                            module.alpha / module.down.weight.shape[0]  # type: ignore
                            if module.alpha
                            else 1.0
                        )
                    )
                    p.weight.copy_(weight.to(p.weight.dtype))
            if skipped:
                logger.warn(f"Broken weight on {getattr(p, 'layer_name', 'UNKNOWN')}.")
            setattr(p, "lora_current_names", wanted_names)
