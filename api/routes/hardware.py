from typing import List

import torch
from fastapi import APIRouter, HTTPException

from core.shared import all_gpus, amd

router = APIRouter(tags=["hardware"])


@router.get("/driver")
async def driver():
    "Return the version of the NVIDIA driver"

    if amd:
        return "1.0.0"
    from pynvml import nvml

    return nvml.nvmlSystemGetDriverVersion()


@router.get("/gpu_ids")
async def gpu_ids() -> List[int]:
    "List all available GPUs"

    if amd:
        import pyamdgpuinfo

        return list(range(pyamdgpuinfo.detect_gpus()))

    return list(range(torch.cuda.device_count()))


@router.get("/gpu_name/{gpu_id}")
async def gpu(gpu_id: int) -> str:
    "Return the name of the GPU"

    if amd:
        return all_gpus[gpu_id].name

    return torch.cuda.get_device_name(gpu_id)


@router.get("/gpu_memory/{gpu_id}")
async def gpu_memory(gpu_id: int):
    "Return the memory statistics of the GPU"

    if amd:
        amdgpu = all_gpus[gpu_id]
        data = amdgpu.memory_info
        return (data["vram_size"], data["vram_size"] - amdgpu.query_vram_usage(), "b")
    else:
        from pynvml import smi

        nvsmi = smi.nvidia_smi.getInstance()
        assert nvsmi is not None

        data = nvsmi.DeviceQuery("memory.free, memory.total")["gpu"]
        try:
            data = data[gpu_id]
            data = data["fb_memory_usage"]
            total = data["total"]
            free = data["free"]
            unit = data["unit"]

            return (total, free, unit)
        except IndexError:
            raise HTTPException(  # pylint: disable=raise-missing-from
                status_code=400, detail="GPU not found"
            )


@router.get("/gpus")
async def gpus():
    "List all available GPUs"

    devices = {}
    for i in await gpu_ids():
        if amd:
            data = all_gpus[i]
            name = data.name
            total_memory = data.memory_info["vram_size"]  # type: ignore
            major = 8
            minor = 1
            multi_processor_count = 1000
        else:
            from torch._C import _CudaDeviceProperties

            data: _CudaDeviceProperties = torch.cuda.get_device_properties(i)
            name = data.name
            total_memory = data.total_memory
            major = data.major
            minor = data.minor
            multi_processor_count = data.multi_processor_count

        devices[i] = {
            "name": name,
            "total_memory": str(round(total_memory / 1024**3)) + "GB",
            "major": major,
            "minor": minor,
            "multi_processor_count": multi_processor_count,
        }

    return devices
