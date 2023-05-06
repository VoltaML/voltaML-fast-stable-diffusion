from typing import Literal, List
import io
import logging
import warnings

import torch
import numpy as np
from torch.utils._pytree import tree_map

from core.config import config
from .trace_utils import generate_inputs


Backend = Literal["llvm-cpu", "vmvx", "cuda", "vulkan"]

logger = logging.getLogger(__name__)


def is_iree_available():
    "Checks if IREE is available"
    try:
        import iree.compiler  # pylint: disable=unused-import
        import iree.runtime

        import torch_mlir  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


class IREEInvoker:
    "Copied from https://github.com/iree-org/iree-torch/blob/main/python/iree_torch/__init__.py"

    def __init__(self, iree_module):
        self._iree_module = iree_module
        self.device = iree_module._context.config.device

    def __getattr__(self, function_name: str):
        import iree.runtime as irt

        def invoke(*args):
            def wrap(x):
                if isinstance(x, torch.Tensor):
                    return irt.asdevicearray(
                        self.device, x.detach().numpy().copy(order="C")
                    )
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
    import iree.compiler as ic

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
    import iree.runtime as irt

    if backend == "cuda" or backend == "vulkan":
        conf = irt.Config(device=irt.get_device(backend))
    else:
        conf = irt.Config(
            driver_name="local-sync" if backend in ("llvm-cpu", "vmvx") else backend,
        )
    ctx = irt.SystemContext(config=conf)
    vm_module = irt.VmModule.from_flatbuffer(ctx.instance, flatbuffer)
    ctx.add_vm_module(vm_module)
    return IREEInvoker(ctx.modules.module)


def convert_pipe_state_to_iree(
    pipe, low_memory_mode: bool = False, use_dynamo: bool = True, ltc: bool = False
):
    "Convert a pipeline to IREE backend. Will break all runtime things (LORAs and Textual Inversion)"
    if not is_iree_available():
        logger.warning(
            'IREE is not available. Please install it by launching main.py with the args "--install-mlir"'
        )
        return

    import torch_mlir as mlir

    if ltc:
        raise NotImplementedError(
            "Lazy Tensor Core will be implemented at a later date."
        )
    else:
        backend = {"llvm": "llvm-cpu", "interpreted": "vmvx"}.get(
            config.api.iree_target, config.api.iree_target
        )
        extra_args = [
            f"--cost-kind={'code-size' if low_memory_mode else 'throughput'}",
            "--iree-opt-const-eval",
            "--iree-opt-const-expr-hoisting",
            "--iree-opt-numeric-precision-reduction",
            "--iree-opt-strip-assertions",
            "--tail-predication=enabled",
            "--sve-tail-folding=recurrences",
            "--instcombine-code-sinking",
            f"--iree-stream-partitioning-favor={'min-peak-memory' if low_memory_mode else 'max-concurrency'}",
        ]
        if backend == "vmvx":
            extra_args += "--iree-vmvx-enable-microkernels"
            extra_args += "--iree-vmvx-enable-microkernels-decompose-linalg-generic"
        elif backend == "llvm-cpu":
            for arg in [
                "--iree-llvmcpu-reassociate-fp-reductions",
                "--iree-llvmcpu-enable-pad-consumer-fusion",
                "--iree-llvmcpu-loop-vectorization",
                "--iree-llvmcpu-loop-unrolling",
                "--iree-llvmcpu-loop-interleaving",
                "--iree-llvmcpu-slp-vectorization",
                "--iree-llvmcpu-target-cpu=host",
                "--iree-llvmcpu-target-cpu-features=host",
                "--iree-llvmcpu-target-float-abi=soft",
                "--iree-codegen-llvmcpu-enable-transform-dialect-jit",
            ]:
                extra_args.append(arg)
        else:
            for arg in [
                "--iree-codegen-llvmgpu-enable-transform-dialect-jit",
                "--iree-codegen-llvmgpu-use-mma-sync",
                "--iree-codegen-gpu-native-math-precision",
                "--iree-codegen-enable-vector-peeling",
                "--iree-codegen-enable-vector-padding",
            ]:
                extra_args.append(arg)

        if use_dynamo:
            from torch_mlir.dynamo import make_simple_dynamo_backend

            @make_simple_dynamo_backend
            def iree_backend(fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):  # type: ignore
                mlir_module = mlir.compile(
                    fx_graph,  # type: ignore
                    example_inputs,
                    output_type=mlir.OutputType.LINALG_ON_TENSORS,
                )
                mlir_module = compile_to_vmfb(mlir_module, target_backend=backend, extra_args=extra_args)  # type: ignore
                mlir_module = load_vmfb(mlir_module, backend=backend)  # type: ignore

                def compiled_callable(*inputs):
                    return mlir_module.forward(*inputs)

                return compiled_callable

            compiled_model = torch.compile(pipe.unet, backend=iree_backend)
            compiled_model(*generate_inputs(pipe.unet.dtype, pipe.unet.device))
            pipe.unet = compiled_model  # type: ignore
        else:
            from .trace_utils import TracedUNet

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlir_module = mlir.compile(
                    pipe.unet,
                    generate_inputs(pipe.unet.dtype, pipe.unet.device),
                    output_type=mlir.OutputType.LINALG_ON_TENSORS,
                    use_tracing=True,
                )
            mlir_module = compile_to_vmfb(mlir_module, target_backend=backend, extra_args=extra_args)  # type: ignore
            mlir_module = load_vmfb(mlir_module, backend=backend)  # type: ignore
            mlir_module.in_channels = pipe.unet.in_channels  # type: ignore pylint: disable=attribute-defined-outside-init
            mlir_module.config = pipe.unet.config  # type: ignore pylint: disable=attribute-defined-outside-init
            mlir_module.dev = pipe.unet.device  # type: ignore pylint: disable=attribute-defined-outside-init
            mlir_module.dtype = pipe.unet.dtype  # type: ignore pylint: disable=attribute-defined-outside-init
            pipe.unet = TracedUNet(mlir_module)
        return pipe.unet
