import logging

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect

from api import websocket_manager
from api.websockets.data import Data

router = APIRouter(tags=["websockets"])
logger = logging.getLogger(__name__)


@router.websocket("/master")
async def master_endpoint(websocket: WebSocket):
    "Main connection point for the websocket"

    await websocket_manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            if text == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@router.post("/progress")
def set_progress(progress: int):
    "Set the progress of the progress bar"

    websocket_manager.broadcast_sync(
        data=Data(data_type="progress", data={"progress": progress})
    )


@router.get("/get-active-connetions")
def get_active_connections():
    connections = websocket_manager.get_active_connections()
    converted_connections = [
        f"{connection.client.host}:{connection.client.port}-{connection.client_state.name}"
        for connection in connections
        if connection.client is not None
    ]

    return converted_connections
