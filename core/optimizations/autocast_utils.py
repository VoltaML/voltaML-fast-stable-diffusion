import contextlib
import importlib
from typing import Any, Optional

import torch

from core.config import config

_initialized_directml = False
_patch_list = [
    "torch.Tensor.__matmul__",
    "torch.addbmm",
    "torch.addmm",
    "torch.addmv",
    "torch.addr",
    "torch.baddbmm",
    "torch.bmm",
    "torch.chain_matmul",
    "torch.linalg.multi_dot",
    "torch.nn.functional.conv1d",
    "torch.nn.functional.conv2d",
    "torch.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose1d",
    "torch.nn.functional.conv_transpose2d",
    "torch.nn.functional.conv_transpose3d",
    "torch.nn.GRUCell",
    "torch.nn.functional.linear",
    "torch.nn.LSTMCell",
    "torch.matmul",
    "torch.mm",
    "torch.mv",
    "torch.prelu",
    "torch.nn.RNNCell",
]


def autocast(
    dtype: torch.dtype,
    disable: bool = False,
):
    "Context manager to autocast tensors to desired dtype for all supported backends"

    global _initialized_directml  # pylint: disable=global-statement

    if dtype == torch.float32 or disable:
        return contextlib.nullcontext()
    if config.api.device_type == "directml":
        if not _initialized_directml:
            for p in _patch_list:
                _patch(p)
            _initialized_directml = True
        return torch.dml.autocast(dtype=dtype, disable=False)  # type: ignore
    if config.api.device_type == "intel":
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)  # type: ignore
    if config.api.device_type == "cpu":
        return torch.cpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)  # type: ignore
    return torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)  # type: ignore


def without_autocast(disable: bool = False):
    "Context manager to disable autocast"

    if disable:
        return contextlib.nullcontext()
    if config.api.device_type == "directml":
        return torch.dml.autocast(disable=True)  # type: ignore
    if config.api.device_type == "intel":
        return torch.xpu.amp.autocast(enabled=False, cache_enabled=False)  # type: ignore
    if config.api.device_type == "cpu":
        return torch.cpu.amp.autocast(enabled=False, cache_enabled=False)  # type: ignore
    return torch.cuda.amp.autocast(enabled=False, cache_enabled=False)  # type: ignore


def _new_forward(forward, args, kwargs):
    if not torch.dml.is_autocast_enabled():  # type: ignore
        return forward(*args, **kwargs)

    def cast(t):
        if not isinstance(t, torch.Tensor):
            return t
        return t.type(torch.dml.get_autocast_dtype())  # type: ignore

    args = list(map(cast, args))
    for kwarg in kwargs:
        kwargs[kwarg] = cast(kwargs[kwarg])
    return forward(*args, **kwargs)


def _patch(imp: str):
    f = imp.split(".")
    for i in range(len(f) - 1, -1, -1):
        try:
            rs = importlib.import_module(".".join(f[:i]))
            break
        except ImportError:
            pass
    for attr in f[i:-1]:  # type: ignore
        rs = getattr(rs, attr)  # type: ignore
    op = getattr(rs, f[-1])  # type: ignore
    setattr(rs, f[-1], lambda *args, **kwargs: _new_forward(op, args, kwargs))  # type: ignore


class dml:
    _autocast_enabled: bool = False
    _autocast_dtype: torch.dtype = torch.float16

    def set_autocast_enabled(value: bool = True):  # type: ignore pylint: disable=no-self-argument
        dml._autocast_enabled = value

    def is_autocast_enabled() -> bool:  # type: ignore pylint: disable=no-method-argument
        return dml._autocast_enabled

    def set_autocast_dtype(dtype: torch.dtype = torch.float16):  # type: ignore pylint: disable=no-self-argument
        dml._autocast_dtype = dtype

    def get_autocast_dtype() -> torch.dtype:  # type: ignore pylint: disable=no-method-argument
        return dml._autocast_dtype

    class autocast:
        def __init__(
            self,
            dtype: Optional[torch.device] = None,
            disable: bool = False,
        ):
            self.prev = self.prev_d = None
            self.dtype = dtype or torch.dml.get_autocast_dtype()  # type: ignore
            self.disable = disable

        def __enter__(self):
            if not self.disable:
                self.prev = torch.dml.is_autocast_enabled()  # type: ignore
                self.prev_d = torch.dml.get_autocast_dtype()  # type: ignore
                torch.dml.set_autocast_enabled(True)  # type: ignore
                torch.dml.set_autocast_dtype(self.dtype)  # type: ignore

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
            torch.dml.set_autocast_enabled(self.prev or False)  # type: ignore
            torch.dml.set_autocast_dtype(self.prev_d or torch.dml.get_autocast_dtype())  # type: ignore


torch.dml = dml  # type: ignore
