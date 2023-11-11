import asyncio
import logging
import mimetypes
import os
from pathlib import Path

from api_analytics.fastapi import Analytics
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi_simple_cachecontrol.middleware import CacheControlMiddleware
from fastapi_simple_cachecontrol.types import CacheControl
from huggingface_hub.hf_api import LocalTokenNotFoundError

from api import websocket_manager
from api.routes import static, ws
from api.websockets.data import Data
from api.websockets.notification import Notification
from core import shared
from core.files import get_full_model_path
from core.types import InferenceBackend
from core.utils import determine_model_type

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


@app.exception_handler(LocalTokenNotFoundError)
async def hf_token_error(_request, _exc):
    await websocket_manager.broadcast(
        data=Data(
            data_type="token",
            data={"huggingface": "missing"},
        )
    )

    return JSONResponse(
        content={
            "status_code": 10422,
            "message": "HuggingFace token not found",
            "data": None,
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


@app.exception_handler(404)
async def custom_http_exception_handler(request: Request, _exc):
    "Redirect back to the main page (frontend will handle it)"

    if request.url.path.startswith("/api"):
        return JSONResponse(
            content={
                "status_code": 10404,
                "message": "Not Found",
                "data": None,
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return FileResponse("frontend/dist/index.html")


@app.on_event("startup")
async def startup_event():
    "Prepare the event loop for other asynchronous tasks"

    if logger.level > logging.DEBUG:
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()

    shared.asyncio_loop = asyncio.get_event_loop()
    websocket_manager.loop = shared.asyncio_loop

    perf_task = asyncio.create_task(websocket_manager.perf_loop())
    shared.asyncio_tasks.append(perf_task)

    from core.config import config

    if config.api.autoloaded_models:
        from core.shared_dependent import cached_model_list, gpu

        all_models = cached_model_list.all()

        for model in config.api.autoloaded_models:
            if model in [i.path for i in all_models]:
                backend: InferenceBackend = [i.backend for i in all_models if i.path == model][0]  # type: ignore
                model_type = determine_model_type(get_full_model_path(model))[1]

                gpu.load_model(model, backend, type=model_type)
            else:
                logger.warning(f"Autoloaded model {model} not found, skipping")

    logger.info("Started WebSocketManager performance monitoring loop")
    logger.info(f"UI Available at: http://localhost:{shared.api_port}/")


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
    logger.debug("No FastAPI Analytics key provided, skipping")

# Mount routers
## HTTP
app.include_router(static.router)

# Walk the routes folder and mount all routers
for file in Path("api/routes").iterdir():
    if file.is_file():
        if (
            file.name != "__init__.py"
            and file.suffix == ".py"
            and file.stem not in ["static", "ws"]
        ):
            logger.debug(f"Mounting: {file} as /api/{file.stem}")
            module = __import__(f"api.routes.{file.stem}", fromlist=["router"])
            app.include_router(module.router, prefix=f"/api/{file.stem}")

## WebSockets
app.include_router(ws.router, prefix="/api/websockets")

# Mount outputs folder
app.mount("/data/outputs", StaticFiles(directory="data/outputs"), name="outputs")

# Mount static files (css, js, images, etc.)
static_app = FastAPI()
static_app.add_middleware(
    CacheControlMiddleware, cache_control=CacheControl("no-cache")
)
static_app.mount("/", StaticFiles(directory="frontend/dist/assets"), name="assets")

app.mount("/assets", static_app)
app.mount("/static", StaticFiles(directory="static"), name="extra_static_files")
app.mount("/themes", StaticFiles(directory="data/themes"), name="themes")

origins = ["*"]

# Allow CORS for specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
static_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
