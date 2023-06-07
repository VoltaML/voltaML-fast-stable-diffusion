from typing import Optional

from diffusers.models.attention import Attention
import torch


class MultiheadAttention(torch.nn.MultiheadAttention):
    "Normal torch multihead attention. Taken once again from @Birch-sans diffusers-play repository. Thank you <3"

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        inner_dim = dim_head * heads
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        super().__init__(
            embed_dim=inner_dim,
            num_heads=heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
        )

    def forward(  # pylint: disable=arguments-differ
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        kv = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
            _, vision_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, vision_tokens, -1)
        out, _ = super().forward(
            query=hidden_states,
            key=kv,
            value=kv,
            need_weights=False,
            attn_mask=attention_mask,
        )
        return out


def apply_multihead_attention(module: torch.nn.Module):
    "Apply torch.nn.MultiheadAttention as attention processor."

    def mha_attn(module: torch.nn.Module) -> None:
        for name, m in module.named_children():
            if isinstance(m, Attention):
                setattr(module, name, _to_mha(m))

    module.apply(mha_attn)


def _to_mha(ca: Attention) -> MultiheadAttention:
    bias = ca.to_k.bias is not None  # type: ignore
    assert bias is False
    mha = MultiheadAttention(
        query_dim=ca.to_q.in_features,
        cross_attention_dim=ca.to_k.in_features,  # type: ignore
        heads=ca.heads,
        dim_head=ca.to_q.out_features // ca.heads,
        dropout=ca.to_out[1].p,  # type: ignore
        bias=bias,
    )
    # is self-attention?
    if ca.to_q.in_features == ca.to_k.in_features:  # type: ignore
        mha.get_parameter("in_proj_weight").data = torch.cat(
            [ca.to_q.weight, ca.to_k.weight, ca.to_v.weight]  # type: ignore
        )
    else:
        mha.get_parameter("q_proj_weight").data = ca.to_q.weight
        mha.get_parameter("k_proj_weight").data = ca.to_k.weight  # type: ignore
        mha.get_parameter("v_proj_weight").data = ca.to_v.weight  # type: ignore
    mha.out_proj.weight = ca.to_out[0].weight  # type: ignore
    mha.out_proj.bias = ca.to_out[0].bias  # type: ignore
    return mha
