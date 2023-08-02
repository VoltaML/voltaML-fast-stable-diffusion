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
# pylint: disable=unused-argument, protected-access

from inspect import isfunction
from typing import Any, Optional

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor, nn
from aitemplate.testing import detect_target

USE_CUDA = detect_target().name() == "cuda"


def default(val, d) -> Any:
    if val is not None:
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale: float = dim_head * -0.5
        self.heads: int = heads
        self.dim_head: int = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout, dtype=dtype)  # type: ignore
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        bs = q.shape()[0]

        q = ops.reshape()(q, [bs, -1, self.heads, self.dim_head])
        k = ops.reshape()(k, [bs, -1, self.heads, self.dim_head])
        v = ops.reshape()(v, [bs, -1, self.heads, self.dim_head])
        q = ops.permute()(q, [0, 2, 1, 3])
        k = ops.permute()(k, [0, 2, 1, 3])
        v = ops.permute()(v, [0, 2, 1, 3])

        attn_op = ops.mem_eff_attention(causal=False)
        if not USE_CUDA:
            attn_op = ops.bmm_softmax_bmm_permute(shape=(self.heads,), scale=self.scale)
        out = attn_op(
            (ops.reshape()(q, [bs, self.heads, -1, self.dim_head])),
            (ops.reshape()(k, [bs, self.heads, -1, self.dim_head])),
            (ops.reshape()(v, [bs, self.heads, -1, self.dim_head])),
        )
        out = ops.reshape()(out, [bs, -1, self.heads * self.dim_head])
        proj = self.to_out(out)
        proj = ops.reshape()(proj, [bs, -1, self.heads * self.dim_head])
        if residual is not None:
            return proj + residual
        return proj


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dtype: str = "float16") -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, specialization="mul", dtype=dtype)
        self.gate = nn.Linear(dim_in, dim_out, specialization="fast_gelu", dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x, self.gate(x))


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim, specialization="fast_gelu", dtype=dtype)
            )
            if not glu
            else GEGLU(dim, inner_dim, dtype=dtype)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout, dtype=dtype),  # type: ignore
            nn.Linear(inner_dim, dim_out, dtype=dtype),
        )

    def forward(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        shape = ops.size()(x)
        x = self.net(x)
        x = ops.reshape()(x, shape)  # type: ignore
        if residual is not None:
            return x + residual
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        only_cross_attention: bool = False,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim if only_cross_attention else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
        )

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)

        if context_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
            )
        else:
            self.attn2 = None
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype)
        self.checkpoint = checkpoint
        self.param = (dim, n_heads, d_head, context_dim, gated_ff, checkpoint)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = self.attn1(
            self.norm1(x),
            residual=x,
            context=context if self.only_cross_attention else None,
        )
        if self.attn1 is not None:
            x = self.attn2(self.norm2(x), context=context, residual=x)  # type: ignore
        x = self.ff(self.norm3(x), residual=x)
        return x


def Normalize(in_channels: int, dtype: str = "float16"):
    return nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype
    )


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype)
        self.use_linear_projection = use_linear_projection

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype)
        else:
            self.proj_in = nn.Conv2dBias(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, dtype=dtype
            )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    only_cross_attention=only_cross_attention,
                    dtype=dtype,
                )
                for d in range(depth)
            ]
        )

        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels, dtype=dtype)
        else:
            self.proj_out = nn.Conv2dBias(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype
            )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        b, h, w, c = x.shape()
        x_in = x
        x = self.norm(x)
        if self.use_linear_projection:
            x = ops.reshape()(x, [b, -1, c])
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = ops.reshape()(x, [b, -1, c])

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.use_linear_projection:
            x = self.proj_out(x)
            x = ops.reshape()(x, [b, h, w, c])
        else:
            x = ops.reshape()(x, [b, h, w, c])
            x = self.proj_out(x)
        return x + x_in


class CLIPAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 16,
        layer_norm_eps: float = 1e-5,
        hidden_dropout_prob: float = 0.0,
        causal: bool = False,
        mask_seq: int = 0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=hidden_dropout_prob,
            has_residual=False,
            causal=causal,
            mask_seq=mask_seq,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
        residual: Optional[Tensor] = None,
    ) -> Tensor:
        if residual is not None:
            return self.attn(hidden_states, residual)
        else:
            return self.attn(hidden_states)


class QuickGELUActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "GELU",
        drop: int = 0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, specialization="gelu")
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        shape = x.shape()
        x = self.fc1(x)
        x = self.fc2(x, residual)
        return ops.reshape()(x, shape)


class CLIPMLPQuickGelu(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation_fn = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x, residual)
        return ops.reshape()(x, x.shape())


class CLIPEncoderLayer(nn.Module):
    ACT_LAYER_TO_CLIP_MLP_MAP = {"gelu": CLIPMLP, "quick_gelu": CLIPMLPQuickGelu}

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        batch_size: int = 1,
        seq_len: int = 16,
        causal: bool = False,
        mask_seq: int = 0,
        act_layer: str = "gelu",
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.CrossAttention(
            hidden_size,
            seq_len,
            seq_len,
            num_attention_heads,
            qkv_bias=True,
            causal=causal,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = self.ACT_LAYER_TO_CLIP_MLP_MAP[act_layer](
            hidden_size, int(hidden_size * mlp_ratio)
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self, hidden_states: Tensor, output_attentions: Optional[bool] = False
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, hidden_states, hidden_states, residual
        )
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_return_dict: bool = False,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        batch_size: int = 1,
        seq_len: int = 64,
        causal: bool = False,
        mask_seq: int = 0,
        act_layer: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    causal=causal,
                    mask_seq=mask_seq,
                    act_layer=act_layer,
                )
                for d in range(num_hidden_layers)
            ]
        )
        self.output_attentions = (output_attentions,)
        self.output_hidden_states = (output_hidden_states,)
        self.use_return_dict = use_return_dict

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        output_attentions = default(output_attentions, self.output_attentions)
        output_hidden_states = default(output_hidden_states, self.output_hidden_states)
        return_dict = default(return_dict, self.use_return_dict)

        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)  # type: ignore
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = hidden_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(
            shape=[vocab_size, hidden_size], dtype=dtype
        )
        self.position_embedding = nn.Embedding(
            shape=[max_position_embeddings, hidden_size], dtype=dtype
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        input_shape = ops.size()(input_ids)

        token_embedding = self.token_embedding.tensor()
        token_embedding = ops.reshape()(
            token_embedding, [1, self.vocab_size, self.embed_dim]
        )
        token_embedding = ops.expand()(token_embedding, [input_shape[0], -1, -1])  # type: ignore

        if inputs_embeds is None:
            inputs_embeds = ops.batch_gather()(token_embedding, input_ids)
        position_embedding = self.position_embedding.tensor()
        position_embedding = ops.reshape()(
            position_embedding, [1, self.max_position_embeddings, self.embed_dim]
        )
        position_embedding = ops.expand()(position_embedding, [input_shape[0], -1, -1])  # type: ignore
        position_embeddings = ops.batch_gather()(position_embedding, position_ids)

        embeddings = inputs_embeds + position_embeddings

        embeddings = ops.reshape()(embeddings, [input_shape[0], input_shape[1], -1])  # type: ignore

        return embeddings


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_return_dict=False,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        batch_size: int = 1,
        seq_len: int = 64,
        causal: bool = False,
        mask_seq: int = 0,
        act_layer: str = "gelu",
    ) -> None:
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(hidden_size=hidden_size)
        self.encoder = CLIPEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            causal=causal,
            mask_seq=mask_seq,
            act_layer=act_layer,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        output_attentions = default(output_attentions, self.output_attentions)
        output_hidden_states = default(output_hidden_states, self.output_hidden_states)
        return_dict = default(return_dict, self.use_return_dict)

        if input_ids is None:
            raise ValueError("input_ids must be specified!")
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        return last_hidden_state
