from typing import Optional, Union

import torch
from torch import Generator as native

from core.config import config

from .philox import PhiloxGenerator


def create_generator(seed: int) -> Union[PhiloxGenerator, torch.Generator]:
    generator = config.api.generator
    if generator == "device" and config.api.overwrite_generator:
        generator = "cpu"
    if generator == "philox":
        return PhiloxGenerator(seed)

    return native(config.api.device if generator == "device" else "cpu").manual_seed(
        seed
    )


def randn(
    shape,
    generator: Union[PhiloxGenerator, torch.Generator],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(generator, PhiloxGenerator):
        return torch.asarray(generator.randn(shape), device=device, dtype=dtype)

    print("randn", generator)

    return torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
        device="cpu" if config.api.overwrite_generator else generator.device,
    ).to(device)


def randn_like(
    x: torch.Tensor,
    generator: Union[PhiloxGenerator, torch.Generator],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return randn(x.shape, generator, device, dtype)
