# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import re
from io import BytesIO
from typing import Dict, Optional, Union
import logging

import requests
import torch
from omegaconf import OmegaConf
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers import (
    StableDiffusionPipeline,  # type: ignore
    StableDiffusionXLPipeline,  # type: ignore
)
from diffusers.models import (
    AutoencoderKL,  # type: ignore
    UNet2DConditionModel,  # type: ignore
)
from diffusers.schedulers import (
    DDIMScheduler,  # type: ignore
    EulerDiscreteScheduler,  # type: ignore
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from core.config import config as volta_config
from core.optimizations.sdxl_unet import UNet2DConditionModel as SDXLUNet2D
from .load import load_checkpoint

logger = logging.getLogger(__name__)


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths,
    checkpoint,
    old_checkpoint,
    attention_paths_to_split=None,
    additional_replacements=None,
    config=None,
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3  # type: ignore

            old_tensor = old_tensor.reshape(
                (num_heads, 3 * channels // num_heads) + old_tensor.shape[1:]
            )
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if (
            attention_paths_to_split is not None
            and new_path in attention_paths_to_split
        ):
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        is_attn_weight = "proj_attn.weight" in new_path or (
            "attentions" in new_path and "to_" in new_path
        )
        shape = old_checkpoint[path["old"]].shape
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int, controlnet=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if controlnet:
        unet_params = original_config.model.params.control_stage_config.params
    else:
        if (
            "unet_config" in original_config.model.params
            and original_config.model.params.unet_config is not None
        ):
            unet_params = original_config.model.params.unet_config.params
        else:
            unet_params = original_config.model.params.network_config.params

    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [
        unet_params.model_channels * mult for mult in unet_params.channel_mult
    ]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnDownBlock2D"
            if resolution in unet_params.attention_resolutions
            else "DownBlock2D"
        )
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnUpBlock2D"
            if resolution in unet_params.attention_resolutions
            else "UpBlock2D"
        )
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params.transformer_depth is not None:
        transformer_layers_per_block = (
            unet_params.transformer_depth
            if isinstance(unet_params.transformer_depth, int)
            else list(unet_params.transformer_depth)
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    head_dim = unet_params.num_heads if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params.use_linear_in_transformer
        if "use_linear_in_transformer" in unet_params
        else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params.model_channels // unet_params.num_head_channels
            head_dim = [head_dim_mult * c for c in list(unet_params.channel_mult)]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params.context_dim is not None:
        context_dim = (
            unet_params.context_dim
            if isinstance(unet_params.context_dim, int)
            else unet_params.context_dim[0]
        )

    if "num_classes" in unet_params:
        if unet_params.num_classes == "sequential":
            if context_dim in [2048, 1280]:
                # SDXL
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                class_embed_type = "projection"
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params.adm_in_channels

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params.in_channels,
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params.num_res_blocks,
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params.disable_self_attentions

    if "num_classes" in unet_params and isinstance(unet_params.num_classes, int):
        config["num_class_embeds"] = unet_params.num_classes

    if controlnet:
        config["conditioning_channels"] = unet_params.hint_channels
    else:
        config["out_channels"] = unet_params.out_channels
        config["up_block_types"] = tuple(up_block_types)

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config


def convert_ldm_unet_checkpoint(
    checkpoint,
    config,
    path=None,
    extract_ema=False,
    skip_extract_state_dict=False,
):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    if skip_extract_state_dict:
        unet_state_dict = checkpoint
    else:
        # extract state_dict for UNet
        unet_state_dict = {}
        keys = list(checkpoint.keys())

        unet_key = "model.diffusion_model."

        # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
        if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
            logger.warning(f"Checkpoint {path} has both EMA and non-EMA weights.")
            logger.warning(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(
                        flat_ema_key
                    )
        else:
            if sum(k.startswith("model_ema") for k in keys) > 100:
                logger.warning(
                    "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                    " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
                )

            for key in keys:
                if key.startswith(unet_key):
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict[
        "time_embed.0.weight"
    ]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict[
        "time_embed.0.bias"
    ]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict[
        "time_embed.2.weight"
    ]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict[
        "time_embed.2.bias"
    ]

    if config["class_embed_type"] is None:
        # No parameters to port
        ...
    elif (
        config["class_embed_type"] == "timestep"
        or config["class_embed_type"] == "projection"
    ):
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict[
            "label_emb.0.0.weight"
        ]
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict[
            "label_emb.0.0.bias"
        ]
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict[
            "label_emb.0.2.weight"
        ]
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict[
            "label_emb.0.2.bias"
        ]
    else:
        raise NotImplementedError(
            f"Not implemented `class_embed_type`: {config['class_embed_type']}"
        )

    if config["addition_embed_type"] == "text_time":
        new_checkpoint["add_embedding.linear_1.weight"] = unet_state_dict[
            "label_emb.0.0.weight"
        ]
        new_checkpoint["add_embedding.linear_1.bias"] = unet_state_dict[
            "label_emb.0.0.bias"
        ]
        new_checkpoint["add_embedding.linear_2.weight"] = unet_state_dict[
            "label_emb.0.2.weight"
        ]
        new_checkpoint["add_embedding.linear_2.bias"] = unet_state_dict[
            "label_emb.0.2.bias"
        ]

    # Relevant to StableDiffusionUpscalePipeline
    if "num_class_embeds" in config:
        if (config["num_class_embeds"] is not None) and (
            "label_emb.weight" in unet_state_dict
        ):
            new_checkpoint["class_embedding.weight"] = unet_state_dict[
                "label_emb.weight"
            ]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "input_blocks" in layer
        }
    )
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "middle_block" in layer
        }
    )
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "output_blocks" in layer
        }
    )
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key
            for key in input_blocks[i]
            if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.weight"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.weight")
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.bias"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            unet_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)

            meta_path = {
                "old": f"input_blocks.{i}.1",
                "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths,
        new_checkpoint,
        unet_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [
                key for key in output_blocks[i] if f"output_blocks.{i}.1" in key
            ]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(
                    ["conv.bias", "conv.weight"]
                )
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.weight"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.weight"]
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.bias"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.bias"]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths,
                    new_checkpoint,
                    unet_state_dict,
                    additional_replacements=[meta_path],
                    config=config,
                )
        else:
            resnet_0_paths = renew_resnet_paths(
                output_block_layers, n_shave_prefix_segments=1
            )
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(
                    [
                        "up_blocks",
                        str(block_id),
                        "resnets",
                        str(layer_in_block_id),
                        path["new"],
                    ]
                )

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = (
        "first_stage_model."
        if any(k.startswith("first_stage_model.") for k in keys)
        else ""
    )
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"
    ]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"
    ]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"
    ]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"
    ]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"
    ]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"
    ]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False, text_encoder=None):
    if text_encoder is None:
        config_name = "openai/clip-vit-large-patch14"
        config = CLIPTextConfig.from_pretrained(
            config_name, local_files_only=local_files_only
        )

        with init_empty_weights():
            text_model = CLIPTextModel(config)  # type: ignore
    else:
        text_model = text_encoder

    keys = list(checkpoint.keys())

    text_model_dict = {}

    remove_prefixes = [
        "cond_stage_model.transformer",
        "conditioner.embedders.0.transformer",
    ]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]

    for param_name, param in text_model_dict.items():
        set_module_tensor_to_device(
            text_model,
            param_name,
            volta_config.api.load_device,  # never set dtype here, it screws things up
            # dtype=volta_config.api.load_dtype,
            value=param,
        )

    return text_model


textenc_conversion_lst = [
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
    ("text_projection", "text_projection.weight"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    (
        "token_embedding.weight",
        "transformer.text_model.embeddings.token_embedding.weight",
    ),
    (
        "positional_embedding",
        "transformer.text_model.embeddings.position_embedding.weight",
    ),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def convert_open_clip_checkpoint(
    checkpoint,
    config_name,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    config = CLIPTextConfig.from_pretrained(
        config_name, **config_kwargs, local_files_only=local_files_only
    )

    with init_empty_weights():
        text_model = (
            CLIPTextModelWithProjection(config)  # type: ignore
            if has_projection
            else CLIPTextModel(config)  # type: ignore
        )

    keys = list(checkpoint.keys())

    keys_to_ignore = []
    if (
        config_name == "stabilityai/stable-diffusion-2"
        and config.num_hidden_layers == 23
    ):
        # make sure to remove all keys > 22
        keys_to_ignore += [
            k
            for k in keys
            if k.startswith("cond_stage_model.model.transformer.resblocks.23")
        ]
        keys_to_ignore += ["cond_stage_model.model.text_projection"]

    text_model_dict = {}

    if prefix + "text_projection" in checkpoint:
        d_model = int(checkpoint[prefix + "text_projection"].shape[0])
    else:
        d_model = 1024

    text_model_dict[
        "text_model.embeddings.position_ids"
    ] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if key in keys_to_ignore:
            continue
        if key[len(prefix) :] in textenc_conversion_map:
            if key.endswith("text_projection"):
                value = checkpoint[key].T.contiguous()
            else:
                value = checkpoint[key]

            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        if key.startswith(prefix + "transformer."):
            new_key = key[len(prefix + "transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][
                    :d_model, :
                ]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][
                    d_model : d_model * 2, :
                ]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][
                    d_model * 2 :, :
                ]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][
                    d_model : d_model * 2
                ]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][
                    d_model * 2 :
                ]
            else:
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )

                text_model_dict[new_key] = checkpoint[key]

    for param_name, param in text_model_dict.items():
        set_module_tensor_to_device(
            text_model,
            param_name,
            volta_config.api.load_device,  # don't set dtype here
            # dtype=volta_config.api.load_dtype,
            value=param,
        )

    return text_model


def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    Returns the image processor and clip image encoder for the img2img unclip pipeline.

    We currently know of two types of stable unclip models which separately use the clip and the openclip image
    encoders.
    """

    image_embedder_config = original_config.model.params.embedder_config

    sd_clip_image_embedder_class = image_embedder_config.target
    sd_clip_image_embedder_class = sd_clip_image_embedder_class.split(".")[-1]

    if sd_clip_image_embedder_class == "ClipImageEmbedder":
        clip_model_name = image_embedder_config.params.model

        if clip_model_name == "ViT-L/14":
            feature_extractor = CLIPImageProcessor()
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
        else:
            raise NotImplementedError(
                f"Unknown CLIP checkpoint name in stable diffusion checkpoint {clip_model_name}"
            )

    elif sd_clip_image_embedder_class == "FrozenOpenCLIPImageEmbedder":
        feature_extractor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=local_files_only
        )
    else:
        raise NotImplementedError(
            f"Unknown CLIP image embedder class in stable diffusion checkpoint {sd_clip_image_embedder_class}"
        )

    return feature_extractor, image_encoder


def download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    original_config_file: str = None,  # type: ignore
    image_size: Optional[int] = None,
    prediction_type: str = None,  # type: ignore
    model_type: str = None,  # type: ignore
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    from_safetensors: bool = False,
    local_files_only=False,
) -> DiffusionPipeline:
    if prediction_type == "v-prediction":
        prediction_type = "v_prediction"

    checkpoint = checkpoint_path_or_dict
    if isinstance(checkpoint, str):
        checkpoint = load_checkpoint(checkpoint, from_safetensors)

    # print(
    #     *list(map(lambda i: str(i[0]) + " " + str(i[1].shape), checkpoint.items())),
    #     sep="\n",
    # )

    # Sometimes models don't have the global_step item
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        logger.debug("global_step key not found in model")
        global_step = None

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]  # type: ignore

    if original_config_file is None:
        key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        key_name_sd_xl_base = (
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        )
        key_name_sd_xl_refiner = (
            "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
        )

        # model_type = "v1"
        config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"

        if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:  # type: ignore
            # model_type = "v2"
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
            if global_step == 110000:
                # v2.1 needs to upcast attention
                upcast_attention = True
        elif key_name_sd_xl_base in checkpoint:
            # only base xl has two text embedders
            config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
        elif key_name_sd_xl_refiner in checkpoint:
            # only refiner xl has embedder and one text embedders
            config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

        original_config_file = BytesIO(requests.get(config_url).content)  # type: ignore

    original_config = OmegaConf.load(original_config_file)

    # Convert the text model.
    if (
        model_type is None
        and "cond_stage_config" in original_config.model.params
        and original_config.model.params.cond_stage_config is not None
    ):
        model_type = original_config.model.params.cond_stage_config.target.split(".")[
            -1
        ]
        logger.debug(f"no `model_type` given, `model_type` inferred as: {model_type}")
    elif model_type is None and original_config.model.params.network_config is not None:
        if original_config.model.params.network_config.params.context_dim == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"
        if image_size is None:
            image_size = 1024
    print(model_type)

    # Check if we have a SDXL or SD model and initialize default pipeline
    pipeline_class = StableDiffusionPipeline  # type: ignore
    if model_type in ["SDXL", "SDXL-Refiner"]:
        pipeline_class = StableDiffusionXLPipeline  # type: ignore

    conv_in_weight = checkpoint.get(  # type: ignore
        "model.diffusion_model.input_blocks.0.0.weight", None
    )
    if conv_in_weight is None:
        num_in_channels = 4
    else:
        num_in_channels = conv_in_weight.shape[1]

    if "unet_config" in original_config.model.params:
        original_config["model"]["params"]["unet_config"]["params"][  # type: ignore
            "in_channels"
        ] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]  # type: ignore
        and original_config["model"]["params"]["parameterization"] == "v"  # type: ignore
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    num_train_timesteps = (
        getattr(original_config.model.params, "timesteps", None) or 1000
    )

    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_dict = {
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "interpolation_type": "linear",
            "num_train_timesteps": num_train_timesteps,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
        }
        scheduler = EulerDiscreteScheduler.from_config(scheduler_dict)
        scheduler_type = "euler"
    else:
        beta_start = getattr(original_config.model.params, "linear_start", None) or 0.02
        beta_end = getattr(original_config.model.params, "linear_end", None) or 0.085
        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)  # type: ignore

    if scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)  # type: ignore
    elif scheduler_type == "ddim":
        scheduler = scheduler

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)  # type: ignore
    unet_config["upcast_attention"] = upcast_attention

    path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=path, extract_ema=extract_ema
    )

    with init_empty_weights():
        if model_type in ["SDXL", "SDXL-Refiner"] and volta_config.api.use_minimal_sdxl_pipeline:
            unet = SDXLUNet2D()
        else:
            unet = UNet2DConditionModel(**unet_config)

    if model_type not in ["SDXL", "SDXL-Refiner"]:  # SBM Delay this.
        for param_name, param in converted_unet_checkpoint.items():
            set_module_tensor_to_device(
                unet,
                param_name,
                volta_config.api.load_device,
                dtype=volta_config.api.load_dtype,
                value=param,
            )

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)  # type: ignore
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    if (
        "model" in original_config
        and "params" in original_config.model
        and "scale_factor" in original_config.model.params
    ):
        vae_scaling_factor = original_config.model.params.scale_factor
    else:
        vae_scaling_factor = 0.18215  # default SD scaling factor

    vae_config["scaling_factor"] = vae_scaling_factor

    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)

    for param_name, param in converted_vae_checkpoint.items():
        set_module_tensor_to_device(
            vae,
            param_name,
            volta_config.api.load_device,
            dtype=volta_config.api.load_dtype,
            value=param,
        )

    if model_type == "FrozenOpenCLIPEmbedder":
        config_name = "stabilityai/stable-diffusion-2"
        config_kwargs = {"subfolder": "text_encoder"}

        text_model = convert_open_clip_checkpoint(
            checkpoint,
            config_name,
            local_files_only=local_files_only,
            **config_kwargs,  # type: ignore
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2",
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        pipe = pipeline_class(  # type: ignore
            vae=vae,
            text_encoder=text_model,  # type: ignore
            tokenizer=tokenizer,
            unet=unet,  # type: ignore
            scheduler=scheduler,  # type: ignore
            safety_checker=None,  # type: ignore
            feature_extractor=None,  # type: ignore
        )
    elif model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(
            checkpoint, local_files_only=local_files_only, text_encoder=None
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", local_files_only=local_files_only
        )

        pipe = pipeline_class(  # type: ignore
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,  # type: ignore
            scheduler=scheduler,  # type: ignore
            safety_checker=None,  # type: ignore
            feature_extractor=None,  # type: ignore
        )
    else:
        is_refiner = model_type == "SDXL-Refiner"

        tokenizer = None
        text_encoder = None
        if not is_refiner:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
            text_encoder = convert_ldm_clip_checkpoint(
                checkpoint, local_files_only=local_files_only
            )

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            pad_token="!",
            local_files_only=local_files_only,
        )

        config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        config_kwargs = {"projection_dim": 1280}
        prefix = (
            "conditioner.embedders.0.model."
            if is_refiner
            else "conditioner.embedders.1.model."
        )

        text_encoder_2 = convert_open_clip_checkpoint(
            checkpoint,
            config_name,
            prefix=prefix,
            has_projection=True,
            local_files_only=local_files_only,
            **config_kwargs,
        )

        for (
            param_name,
            param,
        ) in converted_unet_checkpoint.items():  # SBM Now move model to cpu.
            set_module_tensor_to_device(
                unet,
                param_name,
                volta_config.api.load_device,
                dtype=volta_config.api.load_dtype,
                value=param,
            )
        pipeline_kwargs = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "unet": unet,
            "scheduler": scheduler,
        }

        if is_refiner:
            pipeline_kwargs.update({"requires_aesthetics_score": is_refiner})
        else:
            pipeline_kwargs.update({"force_zeros_for_empty_prompt": False})

        pipe = pipeline_class(**pipeline_kwargs)  # type: ignore

    return pipe  # type: ignore
