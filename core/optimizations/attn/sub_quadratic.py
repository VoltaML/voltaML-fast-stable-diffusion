# thank you @Birch-san for your work :)
from functools import partial
from typing import Optional, NamedTuple, Protocol, List
import math

from diffusers.models.attention import Attention
import torch


def apply_subquadratic_attention(
    module: torch.nn.Module,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    chunk_threshold_bytes: Optional[int] = None,
):
    "Applies subquadratic attention to all the cross-attention modules of the provided module."

    def subquad_attn(module: torch.nn.Module) -> None:
        for m in module.children():
            if isinstance(m, Attention):
                processor = SubQuadraticCrossAttnProcessor(
                    query_chunk_size=query_chunk_size,
                    kv_chunk_size=kv_chunk_size,
                    kv_chunk_size_min=kv_chunk_size_min,
                    chunk_threshold_bytes=chunk_threshold_bytes,
                )

                m.set_processor(processor)  # type: ignore

    module.apply(subquad_attn)


class SubQuadraticCrossAttnProcessor:
    "Taken from @Birch-sans fantastic diffusers-play repository."
    query_chunk_size: int
    kv_chunk_size: Optional[int]
    kv_chunk_size_min: Optional[int]
    chunk_threshold_bytes: Optional[int]

    def __init__(
        self,
        query_chunk_size=1024,
        kv_chunk_size: Optional[int] = None,
        kv_chunk_size_min: Optional[int] = None,
        chunk_threshold_bytes: Optional[int] = None,
    ):
        r"""
        Args:
            query_chunk_size (`int`, *optional*, defaults to `1024`)
            kv_chunk_size (`int`, *optional*, defaults to `None`): if None, sqrt(key_tokens) is used.
            kv_chunk_size_min (`int`, *optional*, defaults to `None`): only considered when `kv_chunk_size is None`. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
            chunk_threshold_bytes (`int`, *optional*, defaults to `None`): if defined: only bother chunking if the self-attn matmul would allocate more bytes than this. whenever we can fit traditional attention into memory: we should prefer to do so, as the unchunked algorithm is faster.
        """
        self.query_chunk_size = query_chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.kv_chunk_size_min = kv_chunk_size_min
        self.chunk_threshold_bytes = chunk_threshold_bytes

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        encoder_hidden_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )

        assert (
            attention_mask is None
        ), "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."
        # I don't know what test case can be used to determine whether softmax is computed at sufficient bit-width,
        # but sub-quadratic attention has a pretty bespoke softmax (defers computation of the denominator) so this needs some thought.
        assert (
            not attn.upcast_softmax or torch.finfo(hidden_states.dtype).bits >= 32
        ), "upcast_softmax was requested, but is not implemented"

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)  # type: ignore
        value = attn.to_v(encoder_hidden_states)  # type: ignore

        query = query.unflatten(-1, (attn.heads, -1)).transpose(1, 2).flatten(end_dim=1)
        key_t = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).flatten(end_dim=1)
        del key
        value = value.unflatten(-1, (attn.heads, -1)).transpose(1, 2).flatten(end_dim=1)

        dtype = query.dtype
        bytes_per_token = torch.finfo(query.dtype).bits // 8
        batch_x_heads, q_tokens, _ = query.shape
        _, _, k_tokens = key_t.shape
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

        query_chunk_size = self.query_chunk_size
        kv_chunk_size = self.kv_chunk_size

        if (
            self.chunk_threshold_bytes is not None
            and qk_matmul_size_bytes <= self.chunk_threshold_bytes
        ):
            # the big matmul fits into our memory limit; do everything in 1 chunk,
            # i.e. send it down the unchunked fast-path
            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens

        hidden_states = efficient_dot_product_attention(
            query,
            key_t,
            value,
            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=self.kv_chunk_size_min,
            use_checkpoint=attn.training,
        )

        hidden_states = hidden_states.to(dtype)

        hidden_states = (
            hidden_states.unflatten(0, (-1, attn.heads))
            .transpose(1, 2)
            .flatten(start_dim=2)
        )

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        return hidden_states


class AttnChunk(NamedTuple):
    exp_values: torch.Tensor
    exp_weights_sum: torch.Tensor
    max_score: torch.Tensor


class SummarizeChunk(Protocol):
    @staticmethod
    def __call__(
        query: torch.Tensor,
        key_t: torch.Tensor,
        value: torch.Tensor,
    ) -> AttnChunk:
        ...


class ComputeQueryChunkAttn(Protocol):
    @staticmethod
    def __call__(
        query: torch.Tensor,
        key_t: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        ...


def _summarize_chunk(
    query: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> AttnChunk:
    attn_weights = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key_t,
        alpha=scale,
        beta=0,
    )
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    exp_weights = torch.exp(attn_weights - max_score)
    exp_values = torch.bmm(exp_weights, value)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)


def _query_chunk_attention(
    query: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
) -> torch.Tensor:
    batch_x_heads, k_channels_per_head, k_tokens = key_t.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = _dynamic_slice(
            key_t,
            (0, 0, chunk_idx),  # type: ignore
            (batch_x_heads, k_channels_per_head, kv_chunk_size),  # type: ignore
        )
        value_chunk = _dynamic_slice(
            value,
            (0, chunk_idx, 0),  # type: ignore
            (batch_x_heads, kv_chunk_size, v_channels_per_head),  # type: ignore
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)  # type: ignore
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


def _get_attention_scores_no_kv_chunking(
    query: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attn_scores = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key_t,
        alpha=scale,
        beta=0,
    )
    attn_probs = attn_scores.softmax(dim=-1)
    del attn_scores
    hidden_states_slice = torch.bmm(attn_probs, value)
    return hidden_states_slice


def efficient_dot_product_attention(
    query: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
):
    """Computes efficient dot-product attention given query, transposed key, and value.
    This is efficient version of attention presented in
    https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
    Args:
      query: queries for calculating attention with shape of
        `[batch * num_heads, tokens, channels_per_head]`.
      key_t: keys for calculating attention with shape of
        `[batch * num_heads, channels_per_head, tokens]`.
      value: values to be used in attention with shape of
        `[batch * num_heads, tokens, channels_per_head]`.
      query_chunk_size: int: query chunks size
      kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
      kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
      use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
    Returns:
      Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
    """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, _, k_tokens = key_t.shape
    scale = q_channels_per_head**-0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> torch.Tensor:
        return _dynamic_slice(
            query,
            (0, chunk_idx, 0),  # type: ignore
            (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head),  # type: ignore
        )

    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = (
        partial(torch.utils.checkpoint.checkpoint, summarize_chunk)  # type: ignore
        if use_checkpoint
        else summarize_chunk
    )
    compute_query_chunk_attn: ComputeQueryChunkAttn = (
        partial(_get_attention_scores_no_kv_chunking, scale=scale)
        if k_tokens <= kv_chunk_size
        else (
            # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
            partial(
                _query_chunk_attention,
                kv_chunk_size=kv_chunk_size,
                summarize_chunk=summarize_chunk,
            )
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key_t=key_t,
            value=value,
        )

    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices
    res = torch.cat(
        [
            compute_query_chunk_attn(
                query=get_query_chunk(i * query_chunk_size),
                key_t=key_t,
                value=value,
            )
            for i in range(math.ceil(q_tokens / query_chunk_size))
        ],
        dim=1,
    )
    return res


def _dynamic_slice(
    x: torch.Tensor,
    starts: List[int],
    sizes: List[int],
) -> torch.Tensor:
    slicing = [slice(start, start + size) for start, size in zip(starts, sizes)]
    return x[slicing]
