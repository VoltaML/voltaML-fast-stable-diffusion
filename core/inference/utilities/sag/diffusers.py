from typing import Callable

import torch

from .sag_utils import sag_masking, pred_epsilon, pred_x0


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
    pred: torch.Tensor = pred_x0(pipe, latent, noise, timestep)
    if cfg > 1:
        cond_attn, _ = store_processor.attention_probs.chunk(2)
        text_embeddings, _ = text_embeddings.chunk(2)
    else:
        cond_attn = store_processor.attention_probs

    eps = pred_epsilon(pipe, latent, noise, timestep)
    degraded: torch.Tensor = sag_masking(pipe, pred, cond_attn, map_size, timestep, eps)

    degraded_prep = call(
        degraded.to(dtype=dtype),
        timestep,
        cond=text_embeddings,
        **additional_kwargs,
    )
    return scale * (noise - degraded_prep)
