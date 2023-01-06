import torch
from fastapi import APIRouter

router = APIRouter(tags=["memory"])


@router.get("/stats")
async def free_memory():
    return torch.cuda.memory_stats()


@router.get("/memory_summary")
async def memory_summary():
    return torch.cuda.memory_summary()


@router.get("/memory_allocated")
async def memory_allocated():
    return torch.cuda.memory_allocated()


@router.get("/memory_reserved")
async def memory_reserved():
    return torch.cuda.memory_reserved()
