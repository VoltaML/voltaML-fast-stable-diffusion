#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch
from transformers import CLIPTextConfig, CLIPTextModel

from ..common import torch_dtype_from_str


def map_clip(pt_mod, device="cuda", dtype="float16"):
    if isinstance(pt_mod, dict):
        if "text_model.encoder.layers.22.layer_norm1.weight" in pt_mod.keys():
            clip_text_config = CLIPTextConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=23,
                projection_dim=512,
                hidden_act="gelu",
            )
        else:
            clip_text_config = CLIPTextConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=12,
                projection_dim=768,
            )
        clip_text_model = CLIPTextModel(clip_text_config)
        pt_mod[
            "text_model.embeddings.position_ids"
        ] = clip_text_model.text_model.embeddings.get_buffer("position_ids")
        clip_text_model.load_state_dict(pt_mod)
        pt_params = dict(clip_text_model.named_parameters())
    else:
        pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr

    return params_ait


def map_controlnet(pt_mod, dim=320, device="cuda", dtype="float16"):
    if not isinstance(pt_mod, dict):
        pt_params = dict(pt_mod.named_parameters())
    else:
        pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr
    params_ait["controlnet_cond_embedding_conv_in_weight"] = torch.nn.functional.pad(
        params_ait["controlnet_cond_embedding_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
    )
    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def map_unet(
    pt_mod, in_channels=None, conv_in_key=None, dim=320, device="cuda", dtype="float16"
):
    if in_channels is not None and conv_in_key is None:
        raise ValueError(
            "conv_in_key must be specified if in_channels is not None for padding"
        )
    if not isinstance(pt_mod, dict):
        pt_params = dict(pt_mod.named_parameters())
    else:
        pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "")
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    if conv_in_key is not None:
        if in_channels is None:
            in_channels = 0
        if in_channels > 0 and in_channels < 4:
            pad_by = 4 - in_channels
        elif in_channels > 4 and in_channels < 8:
            pad_by = 8 - in_channels
        elif in_channels > 8 and in_channels < 12:
            pad_by = 12 - in_channels
        else:
            pad_by = 0
        params_ait[conv_in_key] = torch.functional.F.pad(  # type: ignore
            params_ait[conv_in_key], (0, pad_by, 0, 0, 0, 0, 0, 0)
        )

    params_ait["arange"] = torch.arange(start=0, end=dim // 2, dtype=torch.float32).to(
        device, dtype=torch_dtype_from_str(dtype)
    )

    return params_ait


def map_vae(pt_module, device="cuda", dtype="float16", encoder=False):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_ait = {}
    quant_key = "post_quant" if encoder else "quant"
    vae_key = "decoder" if encoder else "encoder"
    for key, arr in pt_params.items():
        if key.startswith(vae_key):
            continue
        if key.startswith(quant_key):
            continue
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            params_ait[key] = torch.permute(arr, [0, 2, 3, 1]).contiguous()
        elif key.endswith("proj_attn_weight"):
            prefix = key[: -len("proj_attn_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("to_out_0_weight"):
            prefix = key[: -len("to_out_0_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("proj_attn_bias"):
            prefix = key[: -len("proj_attn_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("to_out_0_bias"):
            prefix = key[: -len("to_out_0_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("query_weight"):
            prefix = key[: -len("query_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("to_q_weight"):
            prefix = key[: -len("to_q_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("query_bias"):
            prefix = key[: -len("query_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("to_q_bias"):
            prefix = key[: -len("to_q_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("key_weight"):
            prefix = key[: -len("key_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("key_bias"):
            prefix = key[: -len("key_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("value_weight"):
            prefix = key[: -len("value_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("value_bias"):
            prefix = key[: -len("value_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        elif key.endswith("to_k_weight"):
            prefix = key[: -len("to_k_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("to_v_weight"):
            prefix = key[: -len("to_v_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("to_k_bias"):
            prefix = key[: -len("to_k_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("to_v_bias"):
            prefix = key[: -len("to_v_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        else:
            params_ait[key] = arr
    if encoder:
        params_ait["encoder_conv_in_weight"] = torch.functional.F.pad(  # type: ignore
            params_ait["encoder_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
        )

    return params_ait
