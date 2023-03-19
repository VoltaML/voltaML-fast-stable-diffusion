from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/")
async def index():
    "Main page"

    return FileResponse("frontend/dist/index.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    "Icon of the app"

    return FileResponse("frontend/dist/favicon.ico")
