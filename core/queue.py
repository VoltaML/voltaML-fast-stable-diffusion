import asyncio
import logging
import time
from typing import Callable, List, Literal, Tuple

from PIL.Image import Image

from core.errors import DimensionError, ModelFailedError
from core.model_handler import ModelHandler
from core.thread import ThreadWithReturnValue
from core.types import Txt2ImgQueueEntry


def thread_generate(model_handler: ModelHandler, job: Txt2ImgQueueEntry) -> List[Image]:
    "Run the model for a separate thread"

    return model_handler.generate(job)


def thread_load_model(
    model_handler: ModelHandler,
    model: str,
    backend: Literal["PyTorch", "TensorRT"],
    device: str,
) -> None:
    "Load a model in a separate thread"

    model_handler.load_model(model, backend=backend, device=device)


class Queue:
    "A queue for handling jobs"

    def __init__(self) -> None:
        self.jobs: List[Txt2ImgQueueEntry] = []
        self.running = False
        self.model_handler: ModelHandler = ModelHandler()

    async def load_model(
        self,
        model: str,
        backend: Literal["PyTorch", "TensorRT"],
        device: str,
    ) -> None:
        "Load a model into memory"

        _, err = await self.run_in_thread(
            target=thread_load_model,
            args=(self.model_handler, model, backend, device),
        )

        if err:
            raise err

    async def run_in_thread(self, target: Callable, args):
        "Run a function in a separate thread"

        thread = ThreadWithReturnValue(target=target, args=args)
        thread.start()

        # wait for the thread to finish
        while thread.is_alive():
            await asyncio.sleep(0.1)

        # get the value returned from the thread
        return thread.join()

    async def generate(self, job: Txt2ImgQueueEntry) -> Tuple[List[Image], float]:
        "Add a job to the queue and run it if it is first in the queue, wait otherwise"

        try:
            logging.info(f"Adding job {job.data.id} to queue")

            if job.data.width % 8 != 0 or job.data.height % 8 != 0:
                raise DimensionError("Width and height must be divisible by 8")

            self.jobs.append(job)

            # Wait for the job to be at the front of the queue
            while self.jobs[0] != job:
                await asyncio.sleep(0.1)

            start_time = time.time()

            # create a new thread
            images, exception = await self.run_in_thread(
                target=thread_generate, args=(self.model_handler, job)
            )

            logging.info(f"Job {job.data.id} finished")

            self.jobs.pop(0)

            if exception:
                raise exception

            if images is None:
                raise ModelFailedError("Model failed to generate image")

            deltatime = time.time() - start_time

            return (images, deltatime)

        except Exception as error:
            # Clean up the queue
            if job in self.jobs:
                self.jobs.remove(job)

            raise error
