import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from uvicorn import Server

    from core.inference.pytorch import PyTorchStableDiffusion

amd: bool = False
all_gpus: List = []
current_model: Union["PyTorchStableDiffusion", None] = None
current_steps: int = 50
current_done_steps: int = 0
asyncio_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
interrupt: bool = False
threadpool = ThreadPoolExecutor(max_workers=1)

uvicorn_server: Optional["Server"] = None
uvicorn_loop: Optional[asyncio.AbstractEventLoop] = None
asyncio_tasks: List[asyncio.Task] = []

api_port: int = 5003
