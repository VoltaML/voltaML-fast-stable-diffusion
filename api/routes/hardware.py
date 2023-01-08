import torch
from fastapi import APIRouter
from torch._C import _CudaDeviceProperties

router = APIRouter(tags=["hardware"])


@router.get("/stats")
async def free_memory():
    "Returns a dictionary containing memory statistics for current session"

    return torch.cuda.memory_stats()


@router.get("/memory_summary")
async def memory_summary():
    "Returns a string containing a summary of memory usage, including number of allocated and cached tensors and their sizes"

    return torch.cuda.memory_summary()


@router.get("/memory_allocated")
async def memory_allocated():
    "Returns the total amount of memory currently allocated in gigabytes"

    return torch.cuda.memory_allocated() / 1024**3


@router.get("/memory_reserved")
async def memory_reserved():
    "Returns the total amount of memory currently reserved in gigabytes"

    return torch.cuda.memory_reserved() / 1024**3


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
