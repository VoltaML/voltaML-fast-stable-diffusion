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


from typing import List, Optional, Tuple, Union

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor, nn

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import UNetMidBlock2DCrossAttn, get_down_block, get_up_block


class UNet2DConditionModel(nn.Module):
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, str, str, str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, str, str, str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int, int, int, int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        only_cross_attention: List[bool] = [True, True, True, False],
        conv_in_kernel: int = 3,
        dtype: str = "float16",
        time_embedding_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        transformer_layers_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.center_input_sample = center_input_sample
        self.sample_size = sample_size
        self.time_embedding_dim = time_embedding_dim
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        self.in_channels = in_channels
        if in_channels >= 1 and in_channels <= 4:
            in_channels = 4
        elif in_channels > 4 and in_channels <= 8:
            in_channels = 8
        elif in_channels > 8 and in_channels <= 12:
            in_channels = 12
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2dBias(
            in_channels, block_out_channels[0], 3, 1, conv_in_padding, dtype=dtype
        )
        self.time_proj = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos,
            freq_shift,
            dtype=dtype,
            arange_name="arange",
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, dtype=dtype
        )
        self.class_embed_type = class_embed_type
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(
                [num_class_embeds, time_embed_dim], dtype=dtype
            )
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, dtype=dtype
            )
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(dtype=dtype)
        else:
            self.class_embedding = None

        if addition_embed_type == "text_time":
            self.add_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim,
                time_embed_dim,
                dtype=dtype,  # type: ignore
            )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)  # type: ignore

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],  # type: ignore
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=attention_head_dim[i],  # type: ignore
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                dtype=dtype,
            )
            self.down_blocks.append(down_block)  # type: ignore

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],  # type: ignore
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],  # type: ignore
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            dtype=dtype,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))  # type: ignore
        reversed_attention_head_dim = list(reversed(attention_head_dim))  # type: ignore
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block)  # type: ignore
        )
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=reversed_attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                dtype=dtype,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=norm_eps,
            use_swish=True,
            dtype=dtype,
        )

        self.conv_out = nn.Conv2dBias(
            block_out_channels[0], out_channels, 3, 1, 1, dtype=dtype
        )

    def forward(
        self,
        sample: Tensor,
        timesteps: Tensor,
        encoder_hidden_states: Tensor,
        down_block_residual_0: Optional[Tensor] = None,
        down_block_residual_1: Optional[Tensor] = None,
        down_block_residual_2: Optional[Tensor] = None,
        down_block_residual_3: Optional[Tensor] = None,
        down_block_residual_4: Optional[Tensor] = None,
        down_block_residual_5: Optional[Tensor] = None,
        down_block_residual_6: Optional[Tensor] = None,
        down_block_residual_7: Optional[Tensor] = None,
        down_block_residual_8: Optional[Tensor] = None,
        down_block_residual_9: Optional[Tensor] = None,
        down_block_residual_10: Optional[Tensor] = None,
        down_block_residual_11: Optional[Tensor] = None,
        mid_block_residual: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        add_embeds: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Tensor:
        down_block_additional_residuals = (
            down_block_residual_0,
            down_block_residual_1,
            down_block_residual_2,
            down_block_residual_3,
            down_block_residual_4,
            down_block_residual_5,
            down_block_residual_6,
            down_block_residual_7,
            down_block_residual_8,
            down_block_residual_9,
            down_block_residual_10,
            down_block_residual_11,
        )
        mid_block_additional_residual = mid_block_residual
        if down_block_additional_residuals[0] is None:
            down_block_additional_residuals = None

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = ops.batch_gather()(
                self.class_embedding.weight.tensor(),
                class_labels,  # type: ignore
            )
            emb = emb + class_emb

        if add_embeds is not None:
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        if self.in_channels < 4:
            sample = ops.pad_last_dim(4, 4)(sample)  # type: ignore
        elif self.in_channels > 4 and self.in_channels < 8:
            sample = ops.pad_last_dim(4, 8)(sample)  # type: ignore
        elif self.in_channels > 8 and self.in_channels < 12:
            sample = ops.pad_last_dim(4, 12)(sample)  # type: ignore
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_additional_residual._attrs[  # type: ignore
                    "shape"
                ] = down_block_res_sample._attrs["shape"]
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        if mid_block_additional_residual is not None:
            mid_block_additional_residual._attrs["shape"] = sample._attrs["shape"]
            sample += mid_block_additional_residual

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]  # type: ignore
            down_block_res_samples = down_block_res_samples[
                : -len(
                    upsample_block.resnets
                )  # type: ignore
            ]

            if (
                hasattr(upsample_block, "attentions")
                and upsample_block.attentions is not None
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)
        return sample
