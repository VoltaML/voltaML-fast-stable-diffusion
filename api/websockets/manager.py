import asyncio
import logging
from asyncio import AbstractEventLoop
from typing import Coroutine, List, Optional

from fastapi import WebSocket

from api.websockets.data import Data

logger = logging.getLogger(__name__)


class WebSocketManager:
    "Manages active websocket connections"

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[AbstractEventLoop] = None
        self.to_run: List[Coroutine] = []

    async def sync_loop(self):
        "Infinite loop that runs all coroutines in the to_run list"

        while True:
            for task in self.to_run:
                await task
                self.to_run.remove(task)

            await asyncio.sleep(0.1)

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
            self.to_run.append(connection.send_json(data.to_json()))
