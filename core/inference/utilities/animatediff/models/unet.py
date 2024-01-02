# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin  # type: ignore
from diffusers.utils import BaseOutput  # type: ignore
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from core.config import config
from core.inference.utilities.load import load_checkpoint
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from ..pia.inflate import patch_conv3d
from .resnet import InflatedConv3d, InflatedGroupNorm
from . import MMV2_DIM_KEY


logger = logging.getLogger(__name__)


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (  # type: ignore
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (  # type: ignore
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),  # type: ignore
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        addition_embed_type: Optional[str] = None,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False,
        # Additional
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        motion_module_kwargs={},
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)  # type: ignore

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)  # type: ignore

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],  # type: ignore
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],  # type: ignore
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,  # type: ignore
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],  # type: ignore
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,  # type: ignore
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))  # type: ignore
        only_cross_attention = list(reversed(only_cross_attention))  # type: ignore
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],  # type: ignore
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,  # type: ignore
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> dict:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str, module: torch.nn.Module, processors: dict
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(
                    return_deprecated_lora=True
                )

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor, _remove_lora=False):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(
                        processor.pop(f"{name}.processor"), _remove_lora=_remove_lora
                    )

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = (
            num_slicable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(
            module: torch.nn.Module, slice_size: List[int]
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # support controlnet
        added_cond_kwargs: dict = {},
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        drop_encode_decode: bool = False,
        order: int = 0,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # 1 - default
        # 2 - deepcache
        # 3 - faster-diffusion
        method = 1
        if drop_encode_decode:
            method = 3

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.debug("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config["center_input_sample"]:
            sample = 2 * sample - 1.0  # type: ignore

        # time
        if isinstance(timestep, list):
            timesteps = timestep[0]
        else:
            timesteps = timestep
        if not torch.is_tensor(timesteps) and (not isinstance(timesteps, list)):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif (not isinstance(timesteps, list)) and len(timesteps.shape) == 0:  # type: ignore
            timesteps = timesteps[None].to(sample.device)  # type: ignore

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if (not isinstance(timesteps, list)) and len(timesteps.shape) == 1:  # type: ignore
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])  # type: ignore
        elif isinstance(timesteps, list):
            # timesteps list, such as [981,961,941]
            from core.inference.utilities.faster_diffusion import warp_timestep

            timesteps = warp_timestep(timesteps, sample.shape[0]).to(sample.device)  # type: ignore
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        aug_emb = None

        if self.config["addition_embed_type"] == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config["addition_embed_type"] == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())  # type: ignore
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))  # type: ignore
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)  # type: ignore
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        def downsample(downsample_block, additional_residuals: dict):
            nonlocal sample, emb, encoder_hidden_states, attention_mask, down_intrablock_additional_residuals, down_block_res_samples
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # For t2i-adapter CrossAttnDownBlock2D
                if is_adapter and len(down_intrablock_additional_residuals) > 0:  # type: ignore
                    additional_residuals[
                        "additional_residuals"
                    ] = down_intrablock_additional_residuals.pop(  # type: ignore
                        0
                    )

                sample, res_samples = downsample_block(
                    hidden_states=sample,  # type: ignore
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:  # type: ignore
                    sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

            down_block_res_samples += res_samples

        def upsample(upsample_block, i, length, additional={}):
            nonlocal self, down_block_res_samples, forward_upsample_size, sample, emb, encoder_hidden_states, upsample_size, attention_mask

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[length:]
            down_block_res_samples = down_block_res_samples[:length]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    **additional,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = down_intrablock_additional_residuals is not None

        if method == 1:
            sample = self.conv_in(sample)
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                downsample(downsample_block, {})

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals  # type: ignore
                ):
                    down_block_res_sample = (
                        down_block_res_sample + down_block_additional_residual
                    )
                    new_down_block_res_samples = new_down_block_res_samples + (
                        down_block_res_sample,
                    )

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if (
                    hasattr(self.mid_block, "has_cross_attention")
                    and self.mid_block.has_cross_attention
                ):
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0  # type: ignore
                    and sample.shape == down_intrablock_additional_residuals[0].shape  # type: ignore
                ):
                    sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

            if is_controlnet:
                sample = sample + mid_block_additional_residual  # type: ignore

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                upsample(upsample_block, i, -len(upsample_block.resnets))
        elif method == 2:
            raise NotImplementedError(
                "DeepCache isn't implemented yet, I don't even have a clue how you got this error either..."
            )
        else:
            assert order is not None

            mod = config.api.drop_encode_decode

            cond = order <= 5 or order % 5 == 0
            if isinstance(mod, int):
                cond = order <= 5 or order % mod == 0

            if cond:
                # 2. pre-process
                sample = self.conv_in(sample)

                # 3. down
                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    downsample(downsample_block, {})

                if is_controlnet:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                        down_block_res_samples, down_block_additional_residuals  # type: ignore
                    ):
                        down_block_res_sample = (
                            down_block_res_sample + down_block_additional_residual
                        )
                        new_down_block_res_samples = new_down_block_res_samples + (
                            down_block_res_sample,
                        )

                    down_block_res_samples = new_down_block_res_samples

                # 4. mid
                if self.mid_block is not None:
                    if (
                        hasattr(self.mid_block, "has_cross_attention")
                        and self.mid_block.has_cross_attention
                    ):
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = self.mid_block(sample, emb)

                    # To support T2I-Adapter-XL
                    if (
                        is_adapter
                        and len(down_intrablock_additional_residuals) > 0  # type: ignore
                        and sample.shape == down_intrablock_additional_residuals[0].shape  # type: ignore
                    ):
                        sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

                if is_controlnet:
                    sample = sample + mid_block_additional_residual  # type: ignore

                # 4.5. save features
                setattr(self, "skip_feature", deepcopy(down_block_res_samples))
                setattr(self, "toup_feature", sample.detach().clone())
            else:
                down_block_res_samples = self.skip_feature
                sample = self.toup_feature

            for i, upsample_block in enumerate(self.up_blocks):
                upsample(upsample_block, i, -len(upsample_block.resnets))

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    def convert_to_pia(self, checkpoint_name: str):
        return patch_conv3d(self, checkpoint_name)

    @classmethod
    def from_pretrained_2d(
        cls,
        unet,
        motion_module_path,
        is_sdxl: bool = False,
    ):
        motion_module_path = Path(motion_module_path)
        state_dict = unet.state_dict()

        # load the motion module weights
        if motion_module_path.exists() and motion_module_path.is_file():
            motion_state_dict = load_checkpoint(
                motion_module_path.as_posix(),
                motion_module_path.suffix.lower() == ".safetensors",
            )
        else:
            raise FileNotFoundError(
                f"no motion module weights found in {motion_module_path}"
            )

        # merge the state dicts
        motion_state_dict.update(state_dict)

        # check if we have a v1 or v2 motion module
        motion_up_dim = motion_state_dict[MMV2_DIM_KEY].shape
        unet_additional_kwargs = {
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": False,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 24,
                "temporal_attention_dim_div": 1,
            },
        }
        if motion_up_dim[1] != 24:
            logger.debug("Detected V2 motion module")
            if unet_additional_kwargs:
                motion_module_kwargs = unet_additional_kwargs.pop(
                    "motion_module_kwargs", {}
                )
                motion_module_kwargs[
                    "temporal_position_encoding_max_len"
                ] = motion_up_dim[1]
                unet_additional_kwargs["motion_module_kwargs"] = motion_module_kwargs
            else:
                unet_additional_kwargs = {
                    "motion_module_kwargs": {
                        "temporal_position_encoding_max_len": motion_up_dim[2]
                    }
                }

        unet_config = dict(unet.config)
        unet_config["_class_name"] = cls.__name__
        if is_sdxl:
            unet_config["down_block_types"] = [
                "DownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
            ]
            unet_config["up_block_types"] = [
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "UpBlock3D",
            ]
        else:
            unet_config["down_block_types"] = [
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ]
            unet_config["up_block_types"] = [
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ]
        unet_config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        if unet_additional_kwargs is None:
            unet_additional_kwargs = {}
        model: torch.nn.Module = cls.from_config(unet_config, **unet_additional_kwargs)  # type: ignore

        # load the weights into the model
        m, u = model.load_state_dict(motion_state_dict, strict=False)
        logger.debug(f"### missing keys: {len(m)}; ### unexpected keys: {len(u)};")

        params = [
            p.numel() if "temporal" in n.lower() else 0
            for n, p in model.named_parameters()
        ]
        logger.info(f"Loaded {sum(params) / 1e6}M-parameter motion module")

        return model
