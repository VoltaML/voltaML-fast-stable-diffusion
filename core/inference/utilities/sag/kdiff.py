from typing import Callable

import torch

from .sag_utils import sag_masking


def calculate_sag(
    pipe,
    call: Callable,
    store_processor,
    latent: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.IntTensor,
    map_size: tuple,
    text_embeddings: torch.Tensor,
    scale: float,
    cfg: float,
    dtype: torch.dtype,
    **additional_kwargs,
) -> torch.Tensor:
    pred: torch.Tensor = noise  # noise is already pred_x0 with kdiff
    if cfg > 1:
        cond_attn, _ = store_processor.attention_probs.chunk(2)
        text_embeddings, _ = text_embeddings.chunk(2)
    else:
        cond_attn = store_processor.attention_probs

    degraded: torch.Tensor = sag_masking(pipe, pred, cond_attn, map_size, timestep, 0)

    # messed up the order of these two, spent half an hour looking for problems.
    # Epsilon
    compensation = noise - degraded
    degraded = degraded - (noise - latent)

    degraded_pred = call(
        degraded.to(dtype=dtype),
        timestep,
        cond=text_embeddings,
        **additional_kwargs,
    )
    return (noise - (degraded_pred + compensation)) * scale
