# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin  # type: ignore
from diffusers.models.attention import AdaLayerNorm, Attention, FeedForward  # type: ignore
from diffusers.utils import BaseOutput  # type: ignore
from einops import rearrange, repeat
from torch import Tensor, nn
from torch._dynamo import allow_in_graph as maybe_allow_in_graph


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


@maybe_allow_in_graph
class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True  # type: ignore
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)  # type: ignore
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0  # type: ignore
            )

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,  # type: ignore
                    unet_use_temporal_attention=unet_use_temporal_attention,  # type: ignore
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)  # type: ignore
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0  # type: ignore
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,  # type: ignore
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> tuple[Tensor] | Transformer3DModelOutput:
        # validate input dim
        if hidden_states.dim() != 5:
            raise ValueError(
                f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
            )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # shenanigans for motion module
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(
            encoder_hidden_states, "b n c -> (b f) n c", f=video_length
        )

        # 1. Input
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim
            )
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim
            )
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                encoder_attention_mask=encoder_attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)  # type: ignore


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        unet_use_cross_frame_attention: bool = False,
        unet_use_temporal_attention: bool = False,
        final_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn / SC-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)  # type: ignore
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        if unet_use_cross_frame_attention:
            # this isn't actually implemented anywhere in the AnimateDiff codebase or in Diffusers...
            raise NotImplementedError("SC-Attn is not implemented yet.")
        else:
            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if only_cross_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)  # type: ignore
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Temporal Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            if self.use_ada_layer_norm:
                self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)  # type: ignore
            else:
                self.norm1 = nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine
                )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,  # type: ignore
        video_length=None,
    ):
        # SparseCausal-Attention
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        if self.unet_use_cross_frame_attention:
            cross_attention_kwargs["video_length"] = video_length

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)  # type: ignore
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)  # type: ignore
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # 4. Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
