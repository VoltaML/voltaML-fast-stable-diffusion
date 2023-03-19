from fastapi import APIRouter

router = APIRouter()


@router.get("/alive")
async def test():
    return {"message": "Hello World"}
