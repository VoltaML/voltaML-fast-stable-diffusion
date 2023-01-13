import asyncio
import logging
from typing import List

from fastapi import WebSocket

from api.websockets.data import Data

logger = logging.getLogger(__name__)


class WebSocketManager:
    "Manages active websocket connections"

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop = asyncio.get_event_loop()

    async def connect(self, websocket: WebSocket):
        "Accepts a new websocket connection and adds it to the list of active connections"

        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        "Removes a websocket connection from the list of active connections"

        self.active_connections.remove(websocket)

    async def send_personal_message(self, data: Data, websocket: WebSocket):
        "Sends a data message to a specific websocket connection"

        await websocket.send_json(data.to_json())

    async def broadcast(self, data: Data):
        "Broadcasts data message to all active websocket connections"

        for connection in self.active_connections:
            await connection.send_json(data.to_json())

    def broadcast_sync(self, data: Data):
        "Broadcasts data message to all active websocket connections synchronously"

        for connection in self.active_connections:
            self.loop.run_until_complete(connection.send_json(data.to_json()))
