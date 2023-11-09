# pylint: disable=global-statement

import logging

import torch

from core.config import config

logger = logging.getLogger(__name__)
_module: torch.nn.Module = None  # type: ignore


def unload_all():
    global _module
    if _module is not None:
        _module.cpu()
    _module = None  # type: ignore


def ensure_correct_device(module: torch.nn.Module):
    if hasattr(module, "offload_device"):
        global _module

        if module.__class__.__name__ == _module.__class__.__name__:
            return

        device = getattr(module, "offload_device", config.api.device)
        logger.debug(f"Transferring {module.__class__.__name__} to {str(device)}.")

        if _module is not None:
            logger.debug(f"Transferring {_module.__class__.__name__} to cpu.")
            _module.cpu()

        module.to(device=torch.device(device))
        _module = module
    else:
        logger.debug(f"Don't need to do anything with {module.__class__.__name__}.")


def set_offload(module, device: torch.device):
    setattr(module, "offload_device", device)
