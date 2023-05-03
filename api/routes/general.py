from fastapi import APIRouter

from api import websocket_manager
from api.websockets.notification import Notification
from core import shared

router = APIRouter(tags=["general"])


@router.post("/interrupt")
async def interrupt():
    "Interupt the current job"

    shared.interrupt = True
    return {"message": "Interupted"}


@router.post("/shutdown")
async def shutdown():
    "Shutdown the server"

    from core.config import config
    from core.shared import uvicorn_loop, uvicorn_server

    if config.api.enable_shutdown:
        if uvicorn_server is not None:
            await websocket_manager.broadcast(
                data=Notification(
                    message="Shutting down the server",
                    severity="warning",
                    title="Shutdown",
                )
            )
            uvicorn_server.force_exit = True

        assert uvicorn_server is not None
        await uvicorn_server.shutdown()

        assert uvicorn_loop is not None
        uvicorn_loop.stop()

    else:
        await websocket_manager.broadcast(
            data=Notification(
                message="Shutdown is disabled", severity="error", title="Shutdown"
            )
        )
        return {"message": "Shutdown is disabled"}
