import asyncio
import logging
import mimetypes
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette import status
from starlette.responses import JSONResponse

from api import websocket_manager
from api.routes import general, generate, hardware, models, outputs, static, test, ws
from core import shared

logger = logging.getLogger(__name__)


async def log_request(request: Request):
    "Log all requests"

    logger.debug(
        f"url: {request.url}"
        # f"url: {request.url}, params: {request.query_params}, body: {await request.body()}"
    )


app = FastAPI(
    docs_url="/api/docs", redoc_url="/api/redoc", dependencies=[Depends(log_request)]
)

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    "Output validation errors into debug log for debugging purposes"

    logger.debug(exc)
    content = {
        "status_code": 10422,
        "message": f"{exc}".replace("\n", " ").replace("   ", " "),
        "data": None,
    }
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.exception_handler(404)
async def custom_http_exception_handler(_request, _exc):
    "Redirect back to the main page (frontend will handle it)"

    return RedirectResponse("/")


@app.on_event("startup")
async def startup_event():
    "Prepare the event loop for other asynchronous tasks"

    shared.asyncio_loop = asyncio.get_event_loop()

    asyncio.create_task(websocket_manager.sync_loop())
    logger.info("Started WebSocketManager sync loop")
    asyncio.create_task(websocket_manager.perf_loop())
    logger.info("Started WebSocketManager performance monitoring loop")
    logger.info("UI Avaliable at: http://localhost:5003/")


@app.on_event("shutdown")
async def shutdown_event():
    "Close all WebSocket connections"

    logger.info("Closing all WebSocket connections")
    await websocket_manager.close_all()


# Origins that are allowed to access the API
origins = ["*"]

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
app.include_router(generate.router, prefix="/api/generate")
app.include_router(hardware.router, prefix="/api/hardware")
app.include_router(models.router, prefix="/api/models")
app.include_router(outputs.router, prefix="/api/output")
app.include_router(general.router, prefix="/api/general")
app.include_router(ws.router, prefix="/api/websockets")

# Mount static files (css, js, images, etc.)
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

output_folder = Path("data/outputs")
output_folder.mkdir(exist_ok=True)
app.mount("/data/outputs", StaticFiles(directory="data/outputs"), name="outputs")
