from typing import Union, Optional, Any
import importlib
import contextlib

import torch

from core.config import config

def autocast(dtype: torch.dtype, device: Union[torch.device, str] = config.api.device_type, disable: bool = False):
    if isinstance(device, torch.device):
        device = str(device).lower()
    if dtype == torch.float32 or disable:
        return contextlib.nullcontext()
    if "privateuseone" in device or "directml" in device:
        # User is on DirectML
        if torch.dml is None:
            # Setup torch.dml
            class dml:
                _autocast_enabled: bool = False
                _autocast_dtype: torch.dtype = torch.float16

                def set_autocast_enabled(value: bool = True):
                    _autocast_enabled = value
                
                def is_autocast_enabled() -> bool:
                    return _autocast_enabled
                
                def set_autocast_dtype(dtype: torch.dtype = torch.float16):
                    _autocast_dtype = dtype

                def get_autocast_dtype() -> torch.dtype:
                    return _autocast_dtype
                
                class autocast:
                    def __init__(self, dtype: Optional[torch.device] = None, disable: bool = False):
                        self.dtype = dtype or _autocast_dtypex
                        self.disable = disable
                    
                    def __enter__(self):
                        if not self.disable:
                            self.prev = torch.dml.is_autocast_enabled()
                            self.prev_d = torch.dml.get_autocast_dtype()
                            torch.dml.set_autocast_enabled(True)
                            torch.dml.set_autocast_dtype(self.dtype)
                    
                    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
                        if not self.disable or self.prev_d is not None:
                            torch.dml.set_autocast_enabled(self.prev)
                            torch.dml.set_autocast_dtype(self.prev_d)
                    
            torch.dml = dml
        return torch.dml.autocast(dtype=dtype, disable=disable)
    if "xpu" in device:
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
    return torch.autocast(device)


_patch_list = ["torch.Tensor.__matmul__", "torch.addbmm", "torch.addmm", "torch.addmv", "torch.addr", "torch.baddbmm", "torch.bmm", "torch.chain_matmul", "torch.linalg.multi_dot", "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d", "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d", "torch.nn.GRUCell", "torch.nn.functional.linear", "torch.nn.LSTMCell", "torch.matmul", "torch.mm", "torch.mv", "torch.prelu", "torch.nn.RNNCell"]

def _new_forward(forward, args, kwargs):
    if torch.dml is not None:
        if not torch.dml.is_autocast_enabled():
            return forward(*args, **kwargs)
        def cast(t):
            if not isinstance(t, torch.Tensor):
                return t
            return t.type(torch.dml.get_autocast_dtype())
        args = list(map(cast, args))
        for kwarg in kwargs:
            kwargs[kwarg] = cast(kwargs[kwarg])
        return forward(*args, **kwargs)
    else:
        return forward(*args, **kwargs)

def _patch(imp: str):
    f = imp.split(".")
    for i in range(len(f)-1, -1, -1):
        try:
            rs = importlib.import_module(".".join(f[:i]))
            break
        except ImportError:
            pass
    for attr in f[i:-1]:
        rs = getattr(rs, attr)
    op = getattr(rs, f[-1])
    setattr(rs, f[-1], lambda *args, **kwargs: _new_forward(op, args, kwargs))

for p in _patch_list:
    _patch(p)
