from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import logging

from diffusers.models.lora import LoRACompatibleConv
import torch

from ...config import config
from .lora import LoRAManager
from .lycoris import LyCORISManager
from .utils import HookObject

torch.nn.Linear.old_forward = torch.nn.Linear.forward  # type: ignore
LoRACompatibleConv.old_forward = LoRACompatibleConv.forward  # type: ignore


logger = logging.getLogger(__name__)


def load_lora_utilities(pipe):
    "Reset/redirect Linear and Conv2ds forward to the lora processor"
    if hasattr(pipe, "lora_injector"):
        pipe.lora_injector.change_forwards()
    else:
        torch.nn.Linear.forward = torch.nn.Linear.old_forward  # type: ignore
        LoRACompatibleConv.forward = LoRACompatibleConv.old_forward  # type: ignore


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
        self.pipe = None
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
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if not any(
                        [
                            x in child_module.__class__.__name__
                            for x in ["Linear", "Conv"]
                        ]
                    ):
                        continue

                    lora_name = prefix + "_" + name + "_" + child_name
                    lora_name = lora_name.replace(".", "_")
                    target_modules.append((lora_name, child_module))
        # print("---")
        return target_modules

    def _load_state_dict(self, file: Union[Path, str]) -> Dict[str, torch.nn.Module]:
        if file is not Path:
            file = Path(file)
        if file.suffix == ".safetensors":  # type: ignore
            from safetensors.torch import load_file

            state_dict = load_file(file)
        else:
            state_dict = torch.load(file)  # .bin, .pt, .ckpt...

        if hasattr(self.pipe, "text_encoder_2"):
            logger.debug("Mapping SGM")
            unet_config = self.pipe.unet.config  # type: ignore
            state_dict = self._maybe_map_sgm_blocks_to_diffusers(
                state_dict, unet_config
            )
            state_dict = self._convert_kohya_lora_to_diffusers(state_dict)

        return state_dict  # type: ignore

    @torch.no_grad()
    def apply_weights(self, p) -> None:
        "Apply the modifications of the hooks"
        for m in self.managers:
            m.apply_hooks(p)

    def change_forwards(self):
        "Redirect lora forward to this hook manager"
        d = self

        def lora_forward(self, input):
            d.apply_weights(self)

            return self.old_forward(input)

        def diffusers_lora_forward(
            self, hidden_states: torch.Tensor, scale: float = 1.0
        ):
            d.apply_weights(self)

            return self.old_forward(hidden_states, scale)

        torch.nn.Linear.forward = lora_forward
        LoRACompatibleConv.forward = diffusers_lora_forward

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe"""
        assert len(self.modules) == 0
        self.pipe = pipe
        text_encoder_targets = []
        if hasattr(pipe, "text_encoder_2"):
            text_encoder_targets = (
                text_encoder_targets
                + self._get_target_modules(
                    pipe.text_encoder_2, "lora_te2", ["CLIPAttention", "CLIPMLP"]
                )
                + self._get_target_modules(
                    pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"]
                )
            )
        else:
            text_encoder_targets = text_encoder_targets + self._get_target_modules(
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
        self.dtype = config.api.load_dtype

    # Temporary, TODO: replace this with something sensible
    def apply_lycoris(
        self,
        file: Union[Path, str],
        alpha: Optional[float] = None,  # type: ignore
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
        self,
        file: Union[Path, str],
        alpha: Optional[Union[torch.Tensor, float]] = None,  # type: ignore
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

    def _convert_kohya_lora_to_diffusers(self, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        te2_state_dict = {}
        network_alphas = {}

        # every down weight has a corresponding up weight and potentially an alpha weight
        lora_keys = [k for k in state_dict.keys() if k.endswith("lora_down.weight")]
        for key in lora_keys:
            lora_name, lora_key = key.split(".", 1)
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"

            if lora_name.startswith("lora_unet_"):
                diffusers_name = lora_name.replace("lora_unet_", "").replace("_", ".")

                if "input.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace(
                        "input.blocks", "down.blocks"
                    )

                if "middle.block" in diffusers_name:
                    diffusers_name = diffusers_name.replace("middle.block", "mid.block")
                if "output.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace(
                        "output.blocks", "up.blocks"
                    )

                diffusers_name = diffusers_name.replace("emb.layers", "time.emb.proj")

                # SDXL specificity.
                if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
                    pattern = r"\.\d+(?=\D*$)"
                    diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
                if ".in." in diffusers_name:
                    diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
                if ".out." in diffusers_name:
                    diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
                if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
                    diffusers_name = diffusers_name.replace("op", "conv")
                if "skip" in diffusers_name:
                    diffusers_name = diffusers_name.replace(
                        "skip.connection", "conv.shortcut"
                    )

                # LyCORIS specificity.
                if "time.emb.proj" in diffusers_name:
                    diffusers_name = diffusers_name.replace(
                        "time.emb.proj", "time.emb.proj"
                    )

                # General coverage.
                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace(".", "_")
                        unet_state_dict[
                            diffusers_name + "." + lora_key
                        ] = state_dict.pop(key)
                        unet_state_dict[
                            (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                        ] = state_dict.pop(lora_name_up)
                    elif "ff" in diffusers_name:
                        diffusers_name = diffusers_name.replace(".", "_")
                        unet_state_dict[
                            diffusers_name + "." + lora_key
                        ] = state_dict.pop(key)
                        unet_state_dict[
                            (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                        ] = state_dict.pop(lora_name_up)
                elif any(key in diffusers_name for key in ("proj.in", "proj.out")):
                    diffusers_name = diffusers_name.replace(".", "_")
                    unet_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(
                        key
                    )
                    unet_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)
                else:
                    diffusers_name = diffusers_name.replace(".", "_")
                    unet_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(
                        key
                    )
                    unet_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)

            elif lora_name.startswith("lora_te2_"):
                diffusers_name = key.replace("lora_te2_", "")
                if "self_attn" in diffusers_name:
                    te2_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(
                        key
                    )
                    te2_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(
                        ".lora.", ".lora_linear_layer."
                    )
                    diffusers_name = diffusers_name.replace(".", "_")
                    te2_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(
                        key
                    )
                    te2_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)
            elif lora_name.startswith("lora_te"):
                diffusers_name = "_".join(key.split("_")[2:])
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(key)
                    te_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(
                        ".lora.", ".lora_linear_layer."
                    )
                    diffusers_name = diffusers_name.replace(".", "_")
                    te_state_dict[diffusers_name + "." + lora_key] = state_dict.pop(key)
                    te_state_dict[
                        (diffusers_name + "." + lora_key).replace("_down.", "_up.")
                    ] = state_dict.pop(lora_name_up)
            # Rename the alphas so that they can be mapped appropriately.
            if lora_name_alpha in state_dict:
                alpha = state_dict.pop(lora_name_alpha).item()
                if lora_name_alpha.startswith("lora_unet_"):
                    prefix = "lora_unet_"
                elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
                    prefix = "lora_te_"
                else:
                    prefix = "lora_te2_"
                new_name = prefix + diffusers_name.split("_lora")[0] + ".alpha"  # type: ignore
                network_alphas.update({new_name: alpha})

        unet_state_dict = {
            f"lora_unet_{module_name}": params
            for module_name, params in unet_state_dict.items()
        }
        te_state_dict = {
            f"lora_te_{module_name}": params
            for module_name, params in te_state_dict.items()
        }
        te2_state_dict = (
            {
                f"lora_te2_{module_name}": params
                for module_name, params in te2_state_dict.items()
            }
            if len(te2_state_dict) > 0
            else None
        )
        if te2_state_dict is not None:
            te_state_dict.update(te2_state_dict)

        new_state_dict = {**unet_state_dict, **te_state_dict, **network_alphas}
        return new_state_dict

    def _maybe_map_sgm_blocks_to_diffusers(
        self, state_dict, unet_config, delimiter="_", block_slice_pos=5
    ):
        # 1. get all state_dict_keys
        all_keys = list(state_dict.keys())
        sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

        # 2. check if needs remapping, if not return original dict
        is_in_sgm_format = False
        for key in all_keys:
            if any(p in key for p in sgm_patterns):
                is_in_sgm_format = True
                break

        if not is_in_sgm_format:
            return state_dict

        # 3. Else remap from SGM patterns
        new_state_dict = {}
        inner_block_map = ["resnets", "attentions", "upsamplers"]

        # Retrieves # of down, mid and up blocks
        input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

        for layer in all_keys:
            if "text" in layer:
                new_state_dict[layer] = state_dict.pop(layer)
            else:
                layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
                if sgm_patterns[0] in layer:
                    input_block_ids.add(layer_id)
                elif sgm_patterns[1] in layer:
                    middle_block_ids.add(layer_id)
                elif sgm_patterns[2] in layer:
                    output_block_ids.add(layer_id)
                else:
                    raise ValueError(
                        f"Checkpoint not supported because layer {layer} not supported."
                    )

        input_blocks = {
            layer_id: [
                key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key
            ]
            for layer_id in input_block_ids
        }
        middle_blocks = {
            layer_id: [
                key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key
            ]
            for layer_id in middle_block_ids
        }
        output_blocks = {
            layer_id: [
                key
                for key in state_dict
                if f"output_blocks{delimiter}{layer_id}" in key
            ]
            for layer_id in output_block_ids
        }

        # Rename keys accordingly
        for i in input_block_ids:
            block_id = (i - 1) // (unet_config.layers_per_block + 1)
            layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

            for key in input_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = (
                    inner_block_map[inner_block_id]
                    if "op" not in key
                    else "downsamplers"
                )
                inner_layers_in_block = (
                    str(layer_in_block_id) if "op" not in key else "0"
                )
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in middle_block_ids:
            key_part = None
            if i == 0:
                key_part = [inner_block_map[0], "0"]
            elif i == 1:
                key_part = [inner_block_map[1], "0"]
            elif i == 2:
                key_part = [inner_block_map[0], "1"]
            else:
                raise ValueError(f"Invalid middle block id {i}.")

            for key in middle_blocks[i]:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + key_part
                    + key.split(delimiter)[block_slice_pos:]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in output_block_ids:
            block_id = i // (unet_config.layers_per_block + 1)
            layer_in_block_id = i % (unet_config.layers_per_block + 1)

            for key in output_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = inner_block_map[inner_block_id]
                inner_layers_in_block = (
                    str(layer_in_block_id) if inner_block_id < 2 else "0"
                )
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        if len(state_dict) > 0:
            raise ValueError(
                "At this point all state dict entries have to be converted."
            )

        return new_state_dict

    def remove_lora(self, file: Union[Path, str]):
        """Remove the individual LoRA from the pipe."""
        if not isinstance(file, Path):
            file = Path(file)
        del self.managers[0].containers[file.name]
