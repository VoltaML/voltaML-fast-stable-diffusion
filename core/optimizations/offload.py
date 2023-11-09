# pylint: disable=global-statement

import logging

from accelerate import cpu_offload
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
    if hasattr(module, "v_offload_device"):
        global _module

        if module.__class__.__name__ == _module.__class__.__name__:
            return

        device = getattr(module, "v_offload_device", config.api.device)

        if _module is not None:
            logger.debug(f"Transferring {_module.__class__.__name__} to cpu.")
            _module.cpu()

        logger.debug(f"Transferring {module.__class__.__name__} to {str(device)}.")
        module.to(device=torch.device(device))
        _module = module
    else:
        logger.debug(f"Don't need to do anything with {module.__class__.__name__}.")


def set_offload(module: torch.nn.Module, device: torch.device):
    if config.api.offload == "module":
        cpu_offload(module, device, offload_buffers=len(module._parameters) > 0)
    else:
        setattr(module, "v_offload_device", device)
