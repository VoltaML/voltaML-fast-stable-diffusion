import logging
import sys
from pathlib import Path

from fastapi import APIRouter

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared

router = APIRouter(tags=["general"])
logger = logging.getLogger(__name__)


@router.post("/interrupt")
def interrupt():
    "Interupt the current job"

    shared.interrupt = True
    return {"message": "Interupted"}


@router.post("/shutdown")
def shutdown():
    "Shutdown the server"

    from core.config import config
    from core.shared import uvicorn_loop, uvicorn_server

    if config.api.enable_shutdown:
        if uvicorn_server is not None:
            websocket_manager.broadcast_sync(
                data=Notification(
                    message="Shutting down the server",
                    severity="warning",
                    title="Shutdown",
                )
            )
            for task in shared.asyncio_tasks:
                task.cancel()
            uvicorn_server.force_exit = True
            logger.debug("Setting force_exit to True")

        assert uvicorn_server is not None
        assert uvicorn_loop is not None

        uvicorn_loop.run_in_executor(None, uvicorn_server.shutdown)
        logger.debug("Unicorn server shutdown")
        uvicorn_loop.stop()
        logger.debug("Unicorn loop stopped")

        sys.exit(0)

    else:
        websocket_manager.broadcast_sync(
            data=Notification(
                message="Shutdown is disabled", severity="error", title="Shutdown"
            )
        )
        return {"message": "Shutdown is disabled"}


@router.get("/queue-status")
def queue_status():
    "Get the status of the queue"

    from core.shared_dependent import gpu

    queue = gpu.queue

    return {
        "jobs": queue.jobs,
    }


@router.post("/queue-clear")
def queue_clear():
    "Clear the queue"

    from core.shared_dependent import gpu

    queue = gpu.queue

    queue.clear()

    return {"message": "Queue cleared"}


@router.get("/themes")
def themes():
    "Get all available themes"

    path = Path("data/themes")
    files = []
    for file in path.glob("*.json"):
        files.append(file.stem)

    files.sort()
    return files
