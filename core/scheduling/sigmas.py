import inspect
from logging import getLogger
from typing import Optional

import k_diffusion
import torch

from core.types import SigmaScheduler

from .types import Denoiser

sampling = k_diffusion.sampling
logger = getLogger(__name__)


def build_sigmas(
    steps: int,
    denoiser: Denoiser,
    discard_next_to_last_sigma: bool = False,
    scheduler: Optional[SigmaScheduler] = None,
    custom_rho: Optional[float] = None,
    custom_sigma_min: Optional[float] = None,
    custom_sigma_max: Optional[float] = None,
) -> torch.Tensor:
    "Build sigmas (timesteps) from custom values."
    steps += 1 if discard_next_to_last_sigma else 0

    if scheduler is None or scheduler == "automatic":
        logger.debug("No optional scheduler provided. Using default.")
        sigmas = denoiser.get_sigmas(steps)
    else:
        sigma_min, sigma_max = (denoiser.sigmas[0].item(), denoiser.sigmas[-1].item())  # type: ignore
        rho = None
        if scheduler == "polyexponential":
            rho = 1
        elif scheduler == "karras":
            rho = 7

        sigma_min = custom_sigma_min if custom_sigma_min is not None else sigma_min
        sigma_max = custom_sigma_max if custom_sigma_max is not None else sigma_max
        rho = custom_rho if custom_rho is not None else rho

        arguments = {
            "n": steps,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "rho": rho,
        }

        sigma_func = getattr(sampling, f"get_sigmas_{scheduler}")
        params = inspect.signature(sigma_func).parameters.keys()
        for arg, val in arguments.copy().items():
            if arg not in params or val is None:
                del arguments[arg]

        logger.debug(f"Building sigmas with {arguments}")
        sigmas = getattr(sampling, f"get_sigmas_{scheduler}")(**arguments)

    if discard_next_to_last_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas
