# coding=utf-8
# Copyright 2023, Haofan Wang, Qixun Wang, All rights reserved.
# Modified by Tomáš Novák, 2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file
from transformers.models.clip.modeling_clip import CLIPTextModel


def load_safetensors_loras(
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    lora_path: str,
    alpha_text_encoder=0.5,
    alpha_unet=0.5,
    lore_prefix_unet: str = "lora_unet",
    lora_prefix_text_encoder: str = "lora_te",
):
    # load LoRA weight from .safetensors
    state_dict = load_file(lora_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        is_unet = False

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(lora_prefix_text_encoder + "_")[-1].split("_")
            )
            curr_layer = text_encoder
        else:
            is_unet = True
            layer_infos = key.split(".")[0].split(lore_prefix_unet + "_")[-1].split("_")
            curr_layer = unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)  # type: ignore
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:  # pylint: disable=broad-except
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        alpha = alpha_unet if is_unet else alpha_text_encoder
        device = unet.device if is_unet else text_encoder.device

        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = (
                state_dict[pair_keys[0]]
                .squeeze(3)
                .squeeze(2)
                .to(device=device, dtype=torch.float32)
            )
            weight_down = (
                state_dict[pair_keys[1]]
                .squeeze(3)
                .squeeze(2)
                .to(device=device, dtype=torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(  # type: ignore
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(device=device, dtype=torch.float32)
            weight_down = state_dict[pair_keys[1]].to(
                device=device, dtype=torch.float32
            )
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)  # type: ignore

        # update visited list
        for item in pair_keys:
            visited.append(item)
