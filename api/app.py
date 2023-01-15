import asyncio
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette import status

from api import websocket_manager
from api.routes import hardware, models, static, test, txt2img, ws
from core import shared


async def log_request(request: Request):
    "Log all requests"

    logging.debug(
        f"url: {request.url}, params: {request.query_params}, body: {await request.body()}"
    )


app = FastAPI(
    docs_url="/api/docs", redoc_url="/api/redoc", dependencies=[Depends(log_request)]
)


@app.on_event("startup")
async def startup_event():
    "Prepare the event loop for other asynchronous tasks"

    shared.asyncio_loop = asyncio.get_event_loop()
    asyncio.create_task(websocket_manager.sync_loop())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, _: RequestValidationError):
    "Show more info about validation errors"

    logging.error(
        f"url: {request.url}, params: {request.query_params}, body: {await request.body()}"
    )
    content = {"status_code": 10422, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# Origins that are allowed to access the API
origins = [
    "http://localhost:5173",
    "https://localhost:5173",
]

# Allow CORS for specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(static.router)
app.include_router(test.router, prefix="/api/test")
app.include_router(txt2img.router, prefix="/api/txt2img")
app.include_router(hardware.router, prefix="/api/hardware")
app.include_router(models.router, prefix="/api/models")
app.include_router(ws.router, prefix="/api/websockets")

# Mount static files (css, js, images, etc.)
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
