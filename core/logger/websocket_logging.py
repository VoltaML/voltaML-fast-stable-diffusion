from logging import LogRecord, StreamHandler
from typing import TYPE_CHECKING, Optional

from api import websocket_manager
from api.websockets.data import Data
from core.functions import debounce

if TYPE_CHECKING:
    from core.config.config import Configuration


class WebSocketLoggingHandler(StreamHandler):
    "Broadcasts log messages to all connected clients."

    def __init__(self, config: Optional["Configuration"]):
        super().__init__()
        self.buffer = []
        self.config = config

    def emit(self, record: LogRecord):
        if not self.config:
            return
        if self.config.api.enable_websocket_logging is False:
            return

        msg = f"{record.levelname} {self.format(record)}"
        self.buffer.insert(0, msg)

        # Prevent buffer from growing too large
        if len(self.buffer) > 100:
            self.send()

        self.debounced_send()

    @debounce(0.5)
    def debounced_send(self):
        self.send()

    def send(self):
        msg = "\n".join(self.buffer)
        self.buffer.clear()
        websocket_manager.broadcast_sync(Data(data={"message": msg}, data_type="log"))
