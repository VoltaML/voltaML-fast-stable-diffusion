from fastapi import APIRouter

from api.shared import state

router = APIRouter(tags=["core"])


@router.post("/interupt")
async def stop():
    "Interupt the current job"

    state.interrupt = True
    return {"message": "Interupted"}
