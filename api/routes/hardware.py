import torch
from fastapi import APIRouter, HTTPException
from pynvml import nvml, smi
from torch._C import _CudaDeviceProperties

router = APIRouter(tags=["hardware"])

nvsmi = smi.nvidia_smi.getInstance()
assert nvsmi is not None


@router.get("/driver")
async def driver():
    "Return the version of the NVIDIA driver"

    return nvml.nvmlSystemGetDriverVersion()


@router.get("/gpu_ids")
async def gpu_ids() -> list[int]:
    "List all available GPUs"

    return list(range(torch.cuda.device_count()))


@router.get("/gpu_name/{gpu_id}")
async def gpu(gpu_id: int) -> str:
    "Return the name of the GPU"

    return torch.cuda.get_device_name(gpu_id)


@router.get("/gpu_memory/{gpu_id}")
async def gpu_memory(gpu_id: int):
    "Return the memory statistics of the GPU"

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
    for i in range(torch.cuda.device_count()):
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
