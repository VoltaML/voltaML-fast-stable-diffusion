import asyncio
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from core.inference.pytorch import PyTorchStableDiffusion
    from core.inference.volta_accelerate import TRTModel

current_model: Union["TRTModel", "PyTorchStableDiffusion", None] = None
current_steps: int = 50
current_done_steps: int = 0
asyncio_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
interrupt: bool = False
