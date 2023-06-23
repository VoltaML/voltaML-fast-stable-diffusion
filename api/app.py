import asyncio
import logging
import mimetypes
import os
from pathlib import Path

from api_analytics.fastapi import Analytics
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_simple_cachecontrol.middleware import CacheControlMiddleware
from fastapi_simple_cachecontrol.types import CacheControl
from starlette import status
from starlette.responses import JSONResponse

from api import websocket_manager
from api.routes import (
    general,
    generate,
    hardware,
    models,
    outputs,
    settings,
    static,
    test,
    ws,
)
from api.websockets.notification import Notification
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

    try:
        why = str(exc).split(":")[1].strip()
        await websocket_manager.broadcast(
            data=Notification(
                severity="error",
                message=f"Validation error: {why}",
                title="Validation Error",
            )
        )
    except IndexError:
        logger.debug("Unable to parse validation error, skipping the error broadcast")

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

    # Inject the logger
    from rich.logging import RichHandler

    # Disable duplicate logger
    logging.getLogger("uvicorn").handlers = []

    for logger_ in ("uvicorn.access", "uvicorn.error", "fastapi"):
        l = logging.getLogger(logger_)
        handler = RichHandler(rich_tracebacks=True, show_time=False)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s » %(message)s", datefmt="%H:%M:%S"
            )
        )
        l.handlers = [handler]

    shared.asyncio_loop = asyncio.get_event_loop()

    sync_task = asyncio.create_task(websocket_manager.sync_loop())
    logger.info("Started WebSocketManager sync loop")
    perf_task = asyncio.create_task(websocket_manager.perf_loop())

    shared.asyncio_tasks.append(sync_task)
    shared.asyncio_tasks.append(perf_task)

    logger.info("Started WebSocketManager performance monitoring loop")
    logger.info("UI Available at: http://localhost:5003/")


@app.on_event("shutdown")
async def shutdown_event():
    "Close all WebSocket connections"

    logger.info("Closing all WebSocket connections")
    await websocket_manager.close_all()


# Enable FastAPI Analytics if key is provided
key = os.getenv("FASTAPI_ANALYTICS_KEY")
if key:
    app.add_middleware(Analytics, api_key=key)
    logger.info("Enabled FastAPI Analytics")
else:
    logger.info("No FastAPI Analytics key provided, skipping")

# Mount routers
## HTTP
app.include_router(static.router)
app.include_router(test.router, prefix="/api/test")
app.include_router(generate.router, prefix="/api/generate")
app.include_router(hardware.router, prefix="/api/hardware")
app.include_router(models.router, prefix="/api/models")
app.include_router(outputs.router, prefix="/api/output")
app.include_router(general.router, prefix="/api/general")
app.include_router(settings.router, prefix="/api/settings")

## WebSockets
app.include_router(ws.router, prefix="/api/websockets")

# Mount outputs folder
output_folder = Path("data/outputs")
output_folder.mkdir(exist_ok=True)
app.mount("/data/outputs", StaticFiles(directory="data/outputs"), name="outputs")

# Mount static files (css, js, images, etc.)
static_app = FastAPI()
static_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
static_app.add_middleware(
    CacheControlMiddleware, cache_control=CacheControl("no-cache")
)
static_app.mount("/", StaticFiles(directory="frontend/dist/assets"), name="assets")

app.mount("/assets", static_app)

# Allow CORS for specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
