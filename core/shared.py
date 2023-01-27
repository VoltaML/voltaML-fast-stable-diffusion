import asyncio
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from core.inference.pytorch import PyTorchInferenceModel
    from core.inference.volta_accelerate import DemoDiffusion

current_model: Union["DemoDiffusion", "PyTorchInferenceModel", None] = None
current_steps: int = 50
asyncio_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
image_decode_steps: int = 5
interrupt: bool = False
