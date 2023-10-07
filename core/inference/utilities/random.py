from typing import Union, Optional

import torch
from torch import Generator as native

from core.config import config
from .philox import Generator as philox

_rng: Union[None, philox, torch.Generator] = None


def create_generator(seed: int) -> None:
    global _rng
    generator = config.api.generator
    if generator == "device" and config.api.overwrite_generator:
        generator = "cpu"
    if generator == "philox":
        _rng = philox(seed)
    else:
        _rng = native(
            config.api.device if generator == "device" else "cpu"
        ).manual_seed(seed)


def randn(
    shape, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert _rng is not None
    if isinstance(_rng, philox):
        return torch.asarray(_rng.randn(shape), device=device, dtype=dtype)
    assert _rng is native
    return torch.randn(
        shape,
        generator=_rng,
        dtype=dtype,
        device="cpu" if config.api.overwrite_generator else _rng.device,
    ).to(device)


def randn_like(
    x: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return randn(x.shape, device, dtype)
