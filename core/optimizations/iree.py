from typing import Literal, List
import io
import platform

import torch
import numpy as np
from torch.utils._pytree import tree_map

from core.config import config
from .trace_utils import TracedUNet, generate_inputs

if platform.system() == "Windows":
    import torch_mlir as mlir

    import iree.runtime as irt
    import iree.compiler as ic



Backend = Literal["llvm-cpu", "vmvx", "cuda", "vulkan"]


class IREEInvoker:
    "Copied from https://github.com/iree-org/iree-torch/blob/main/python/iree_torch/__init__.py"

    def __init__(self, iree_module):
        self._iree_module = iree_module
        self.device = iree_module._context.config.device

    def __getattr__(self, function_name: str):
        def invoke(*args):
            def wrap(x):
                if isinstance(x, torch.Tensor):
                    return irt.asdevicearray(self.device, x)
                return x

            def unwrap(x):
                if isinstance(x, irt.DeviceArray):
                    return torch.from_numpy(np.asarray(x).copy())
                    # return torch.from_numpy(np.asarray(x))
                return x

            iree_args = tree_map(wrap, args)
            result = self._iree_module[function_name](*iree_args)
            return tree_map(unwrap, result)

        return invoke


def compile_to_vmfb(  # pylint: disable=dangerous-default-value
    mlir_module,
    target_backend: Backend = "llvm-cpu",
    extra_args: List[str] = [],
) -> bytes:
    "torch-mlir -> flatbuffer"
    bytecode_stream = io.BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    return ic.compile_str(
        bytecode,
        target_backends=[target_backend],
        input_type=ic.InputType.TM_TENSOR,
        extra_args=extra_args,
    )  # type: ignore


def load_vmfb(flatbuffer, backend: Backend = "llvm-cpu") -> IREEInvoker:
    "Load an IREE flatbuffer"
    conf = irt.Config(
        driver_name="local-sync" if backend in ("llvm-cpu", "vmvx") else backend
    )
    ctx = irt.SystemContext(config=conf)
    vm_module = irt.VmModule.from_flatbuffer(ctx.instance, flatbuffer)
    ctx.add_vm_module(vm_module)
    return IREEInvoker(ctx.modules.module)


def convert_pipe_state_to_iree(pipe, ltc: bool = False):
    "Convert a pipeline to IREE backend. Will break all runtime things (LORAs and Textual Inversion)"
    if platform.system() != "Windows":
        return
    if ltc:
        raise NotImplementedError(
            "Lazy Tensor Core will be implemented at a later date."
        )

    def _nf(latent_model_input, t, encoder_hidden_states):
        return pipe.unet.og_f(
            latent_model_input, t, encoder_hidden_states, return_dict=False
        )[0]

    pipe.unet.og_f = pipe.unet.forward  # pylint: disable=attribute-defined-outside-init
    pipe.unet.forward = _nf
    compiled_model = mlir.compile(
        pipe.unet,
        generate_inputs(dtype=pipe.unet.dtype, device=pipe.unet.device),
        output_type=mlir.OutputType.LINALG_ON_TENSORS,
    )
    backend = {"llvm": "llvm-cpu", "interpreted": "vmvx"}.get(
        config.api.iree_target, config.api.iree_target
    )
    vmfb = compile_to_vmfb(
        compiled_model,
        backend,  # type: ignore
        extra_args=[
            "--iree-opt-const-eval",
            "--iree-opt-const-expr-hoisting",
            "--iree-opt-numeric-precision-reduction",
            "--iree-opt-numeric-precision-reduction",
        ],
    )
    vmfb = load_vmfb(compiled_model, backend)  # type: ignore
    vmfb.in_channels = pipe.unet.in_channels  # type: ignore pylint: disable=attribute-defined-outside-init
    vmfb.dev = pipe.unet.device  # type: ignore pylint: disable=attribute-defined-outside-init
    vmfb.dtype = pipe.unet.dtype  # type: ignore pylint: disable=attribute-defined-outside-init
    vmfb.config = pipe.unet.config  # type: ignore pylint: disable=attribute-defined-outside-init

    def _nv(latent_model_input, t, encoder_hidden_states):
        return [vmfb.og_f(latent_model_input, t, encoder_hidden_states)]

    vmfb.og_f = vmfb.forward  # type: ignore pylint: disable=attribute-defined-outside-init
    vmfb.forward = _nv  # type: ignore pylint: disable=attribute-defined-outside-init

    pipe.unet = TracedUNet(vmfb)  # type: ignore
