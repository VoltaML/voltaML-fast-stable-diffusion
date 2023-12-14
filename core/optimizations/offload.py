# pylint: disable=global-statement

from typing import Optional
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
    global _module
    if _module is not None:
        if module.__class__.__name__ == _module.__class__.__name__:
            return
        logger.debug(f"Transferring {_module.__class__.__name__} to cpu.")
        _module.cpu()
        _module = None  # type: ignore
    if hasattr(module, "v_offload_device"):
        device = getattr(module, "v_offload_device", config.api.device)

        logger.debug(f"Transferring {module.__class__.__name__} to {str(device)}.")
        module.to(device=torch.device(device))
        _module = module
    else:
        logger.debug(f"Don't need to do anything with {module.__class__.__name__}.")


def set_offload(
    module: torch.nn.Module, device: torch.device, offload_type: Optional[str] = None
):
    offload = offload_type or config.api.offload
    if offload == "module":
        class_name = module.__class__.__name__
        if "CLIP" not in class_name and "Autoencoder" not in class_name:
            return cpu_offload(
                module, device, offload_buffers=len(module._parameters) > 0
            )
    setattr(module, "v_offload_device", device)
    return module
