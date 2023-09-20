import torch
from typing import Optional
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from diffusers.models.attention import Attention

def apply_flash_attention(module: torch.nn.Module):
    cross_attn = FlashAttentionStandardAttention()
    self_attn = FlashAttentionQkvAttention()
    def set(mod: torch.nn.Module) -> None:
        if isinstance(mod, Attention):
            if mod.to_k.in_features == mod.to_q.in_features:
                mod.to_qkv = torch.nn.Linear(mod.to_q.in_features, mod.to_q.out_features*3, dtype=mod.to_q.weight.dtype, device=mod.to_q.weight.data.device)
                mod.to_qkv.weight.data = torch.cat([mod.to_q.weight, mod.to_k.weight, mod.to_v.weight]).detach()
                del mod.to_q, mod.to_k, mod.to_v
                mod.set_processor(self_attn)
            else:
                mod.set_processor(cross_attn)
    module.apply(set)

class FlashAttentionBaseAttention:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(attn, hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = self.do(attn, hidden_states, encoder_hidden_states, query)
        hidden_states = hidden_states.flatten(-2)

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        if attn.rescale_output_factor != 1:
            hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def do(self, attn: Attention, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, query: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        "Implemented in classes extending this."
        return None  # type: ignore

    def to_q(self, attn: Attention, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        "Implemented in classes extending this."
        return None  # type: ignore

class FlashAttentionStandardAttention(FlashAttentionBaseAttention):
    # TODO: documentation

    def do(self, attn: Attention, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, query: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        key = attn.to_k(encoder_hidden_states)  # type: ignore
        value = attn.to_v(encoder_hidden_states)  # type: ignore
        query = query.unflatten(-1, (attn.heads, -1))  # type: ignore
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        hidden_states = flash_attn_func(
            query, key, value, dropout_p=0.0, causal=False  # type: ignore
        )
        return hidden_states.to(query.dtype)  # type: ignore
    
    def to_q(self, attn: Attention, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return attn.to_q(hidden_states)  # type: ignore

class FlashAttentionQkvAttention(FlashAttentionBaseAttention):
    # TODO: documentation

    def do(self, attn: Attention, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, query: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        qkv = attn.to_qkv(hidden_states)  # type: ignore
        qkv = qkv.unflatten(-1, (3, attn.heads, -1))
        hidden_states = flash_attn_qkvpacked_func(
            qkv, dropout_p=0.0, causal=False
        )
        return hidden_states.to(qkv.dtype)  # type: ignore