import torch

from .utils import LoRAHookInjector


def load_lora_utilities(pipe):
    if hasattr(pipe, "lora_injector"):
        pipe.lora_injector.change_forwards()
    else:
        torch.nn.Linear.forward = torch.nn.Linear.old_forward  # type: ignore
        torch.nn.Conv2d.forward = torch.nn.Conv2d.old_forward  # type: ignore


def install_lora_hook(pipe):
    """Install LoRAHook to the pipe."""
    if hasattr(pipe, "lora_injector"):
        return
    injector = LoRAHookInjector()
    injector.install_hooks(pipe)
    pipe.lora_injector = injector
    pipe.apply_lora = injector.apply_lora
    pipe.remove_lora = injector.remove_lora


def uninstall_lora_hook(pipe):
    """Uninstall LoRAHook from the pipe."""
    pipe.lora_injector.uninstall_hooks()
    del pipe.lora_injector
    del pipe.apply_lora
    del pipe.remove_lora
