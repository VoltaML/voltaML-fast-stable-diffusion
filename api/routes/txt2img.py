from fastapi import APIRouter

from api.shared import state

router = APIRouter()


@router.post("/interupt")
async def stop():
    state.interrupt = True
    return {"message": "Interupted"}
