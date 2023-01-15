import asyncio
from typing import TYPE_CHECKING, Optional, Union

from core.inference.pytorch import PyTorchInferenceModel

if TYPE_CHECKING:
    from core.inference.volta_accelerate import DemoDiffusion


current_model: Union["DemoDiffusion", PyTorchInferenceModel, None] = None
current_steps: int = 50
asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
