from typing import Callable

import torch

from core.scheduling import KdiffusionSchedulerAdapter
from .cross_attn import CrossAttnStoreProcessor
from .sag_utils import pred_epsilon, pred_x0, sag_masking
from .kdiff import calculate_sag as kdiff
from .diffusers import calculate_sag as diff


def calculate_sag(
    pipe,
    call: Callable,
    store_processor,
    latent: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    timestep: torch.IntTensor,
    map_size: tuple,
    text_embeddings: torch.Tensor,
    scale: float,
    cfg: float,
    dtype: torch.dtype,
    **additional_kwargs,
) -> torch.Tensor:
    if isinstance(pipe.scheduler, KdiffusionSchedulerAdapter):
        return kdiff(
            pipe,
            call,
            store_processor,
            latent,
            noise_pred_uncond,
            timestep,
            map_size,
            text_embeddings,
            scale,
            cfg,
            dtype,
            **additional_kwargs,
        )
    else:
        return diff(
            pipe,
            call,
            store_processor,
            latent,
            noise_pred_uncond,
            timestep,
            map_size,
            text_embeddings,
            scale,
            cfg,
            dtype,
            **additional_kwargs,
        )
