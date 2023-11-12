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
    new_kwargs = {}
    for kw, arg in additional_kwargs.items():
        if arg is not None and isinstance(arg, torch.Tensor):
            if arg.shape[0] != 1:
                arg, _ = arg.chunk(2)
        new_kwargs[kw] = arg

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
            **new_kwargs,
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
            **new_kwargs,
        )
