import tensorrt as trt
import torch
from torch import autocast

trt.init_libnvinfer_plugins(None, "")
from time import time

import numpy as np
import pycuda.autoinit  # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
import pycuda.driver as cuda


class TRTModel:
    """
    Generic class to run a TRT engine by specifying engine path and giving input data.
    """

    class HostDeviceMem(object):
        """
        Helper class to record host-device memory pointer pairs
        """

        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # load and deserialize TRT engine
        self.engine = self.load_engine()

        # allocate input/output memory buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(
            self.engine
        )
        # import ipdb; ipdb.set_trace()
        # create context
        self.context = self.engine.create_execution_context()

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists). From: https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
        self.numpy_to_torch_dtype_dict = {
            bool: torch.bool,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
        }

    def load_engine(self):
        with open(self.engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self, engine):
        """
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:  # binding is the name of input/output
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(
                size, dtype
            )  # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. When cast to int, it's a linear index into the context's memory (like memory address). See https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, model_inputs: list, timing=False):
        """
        Inference step (like forward() in PyTorch).
        model_inputs: list of numpy array or list of torch.Tensor (on GPU)
        """
        # import ipdb; ipdb.set_trace()
        NUMPY = False
        TORCH = False
        if isinstance(model_inputs[0], np.ndarray):
            NUMPY = True
        elif torch.is_tensor(model_inputs[0]):
            TORCH = True
        else:
            assert False, "Unsupported input data format!"

        # batch size consistency check
        if NUMPY:
            batch_size = np.unique(np.array([i.shape[0] for i in model_inputs]))
        elif TORCH:
            batch_size = np.unique(np.array([i.size(dim=0) for i in model_inputs]))
        # assert len(batch_size) == 1, 'Input batch sizes are not consistent!'
        batch_size = batch_size[0]

        for i, model_input in enumerate(model_inputs):
            # print("set input for ",i)
            binding_name = self.engine[i]  # i-th input/output name
            # print("set input for ",binding_name)
            binding_dtype = trt.nptype(
                self.engine.get_binding_dtype(binding_name)
            )  # trt can only tell to numpy dtype
            # print("set input for ",binding_name,binding_dtype)
            # input type cast
            if NUMPY:
                model_input = model_input.astype(binding_dtype)
            elif TORCH:
                model_input = model_input.to(
                    self.numpy_to_torch_dtype_dict[binding_dtype]
                )

            if NUMPY:
                # fill host memory with flattened input data
                np.copyto(self.inputs[i].host, model_input.ravel())
            elif TORCH:
                if timing:
                    cuda.memcpy_dtod(
                        self.inputs[i].device,
                        model_input.data_ptr(),
                        model_input.element_size() * model_input.nelement(),
                    )
                else:
                    # for Torch GPU tensor it's easier, can just do Device to Device copy
                    cuda.memcpy_dtod_async(
                        self.inputs[i].device,
                        model_input.data_ptr(),
                        model_input.element_size() * model_input.nelement(),
                        self.stream,
                    )  # dtod need size in bytes

        if NUMPY:
            if timing:
                [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
            else:
                # input, Host to Device
                [
                    cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
                    for inp in self.inputs
                ]

        duration = 0
        if timing:
            start_time = time()
            self.context.execute_v2(bindings=self.bindings)
            end_time = time()
            duration = end_time - start_time
        else:
            # run inference
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )  # v2 no need for batch_size arg

        if timing:
            [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]
        else:
            # output, Device to Host
            [
                cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
                for out in self.outputs
            ]

        if not timing:
            # synchronize to ensure completion of async calls
            self.stream.synchronize()

        if NUMPY:
            return [out.host.reshape(batch_size, -1) for out in self.outputs], duration
        elif TORCH:
            return [
                torch.from_numpy(out.host.reshape(batch_size, -1))
                for out in self.outputs
            ], duration
