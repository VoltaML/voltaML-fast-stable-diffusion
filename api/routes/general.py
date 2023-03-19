from fastapi import APIRouter

from core import shared

router = APIRouter(tags=["general"])


@router.post("/interrupt")
async def interrupt():
    "Interupt the current job"

    shared.interrupt = True
    return {"message": "Interupted"}
