import asyncio
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from core.inference.pytorch import PyTorchModel
    from core.inference.volta_accelerate import TRTModel

current_model: Union["TRTModel", "PyTorchModel", None] = None
current_steps: int = 50
current_done_steps: int = 0
asyncio_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
image_decode_steps: int = 5
interrupt: bool = False
