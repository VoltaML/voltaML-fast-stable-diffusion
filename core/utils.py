import asyncio
import base64
from io import BytesIO
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, Union

from PIL.Image import Image

from core.thread import ThreadWithReturnValue


def convert_image_to_stream(image: Image) -> BytesIO:
    "Convert an image to a stream of bytes"

    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def convert_image_to_base64(image: Image) -> str:
    "Convert an image to a base64 string"

    stream = convert_image_to_stream(image)
    return base64.b64encode(stream.read()).decode("utf-8")


def convert_base64_to_bytes(data: str):
    "Convert a base64 string to bytes"

    return BytesIO(base64.b64decode(data))


async def run_in_thread_async(
    func: Union[Callable[..., Any], Coroutine[Any, Any, Any]],
    args: Optional[Tuple] = None,
    kwarkgs: Optional[Dict] = None,
) -> Tuple[Union[Any, None], Union[Exception, None]]:
    "Run a function in a separate thread"

    thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwarkgs)
    thread.start()

    # wait for the thread to finish
    while thread.is_alive():
        await asyncio.sleep(0.1)

    # get the value returned from the thread
    return thread.join()
