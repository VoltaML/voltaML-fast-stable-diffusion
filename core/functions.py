import logging

import torch

from api import websocket_manager
from api.websockets.data import Data
from core import shared

logger = logging.getLogger(__name__)


def pytorch_callback(data: dict):
    "Send a websocket message to the client with the progress percentage and partial image"

    _x: torch.Tensor = data["x"]
    step = int(data["i"]) + 1

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="txt2img",
            data={
                "progress": int((step / shared.current_steps) * 100),
                "current_step": step,
                "total_steps": shared.current_steps,
            },
        )
    )
