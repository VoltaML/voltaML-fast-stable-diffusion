import torch

from core.config import config

_module: torch.nn.Module = None  # type: ignore


def unload_all():
    global _module
    if _module is not None:
        _module.cpu()
    _module = None  # type: ignore


def ensure_correct_device(module: torch.nn.Module):
    if hasattr(module, "offload_device"):
        global _module

        if _module is not None:
            _module.cpu()
        module.to(device=getattr(module, "offload_device", config.api.device))
        _module = module


def set_offload(module, device: torch.device):
    setattr(module, "offload_device", device)