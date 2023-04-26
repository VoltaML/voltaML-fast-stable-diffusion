import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Union, List

if TYPE_CHECKING:
    from core.inference.pytorch import PyTorchStableDiffusion
    from core.tensorrt.volta_accelerate import TRTModel

amd: bool = False
all_gpus: List = []
current_model: Union["TRTModel", "PyTorchStableDiffusion", None] = None
current_steps: int = 50
current_done_steps: int = 0
asyncio_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
interrupt: bool = False
hf_token = ""
threadpool = ThreadPoolExecutor(max_workers=1)
