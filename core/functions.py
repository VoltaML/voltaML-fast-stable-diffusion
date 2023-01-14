import torch

from api import websocket_manager
from api.websockets.data import Data
from core import shared


def pytorch_callback(step: int, _timestep: int, _tensor: torch.FloatTensor):
    "Send a websocket message to the client with the progress percentage and partial image"

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
