import asyncio
import base64
import logging
import math
import re
from io import BytesIO
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from PIL import Image

from core.thread import ThreadWithReturnValue

logger = logging.getLogger(__name__)


def get_grid_dimension(length: int) -> Tuple[int, int]:
    "Generate the dimensions of a grid so that images can be tiled"

    cols = math.ceil(length**0.5)
    rows = math.ceil(length / cols)
    return cols, rows


def convert_image_to_stream(image: Image.Image) -> BytesIO:
    "Convert an image to a stream of bytes"

    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def convert_to_image(image: Union[Image.Image, bytes, str]) -> Image.Image:
    "Converts the image to a PIL Image if it is a base64 string or bytes"

    if isinstance(image, str):
        b = convert_base64_to_bytes(image)
        return Image.open(b)

    if isinstance(image, bytes):
        return Image.open(image)

    return image


def convert_image_to_base64(image: Image.Image) -> str:
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
) -> Any:
    "Run a function in a separate thread"

    thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwarkgs)
    thread.start()

    # wait for the thread to finish
    while thread.is_alive():
        await asyncio.sleep(0.1)

    # get the value returned from the thread
    value, exc = thread.join()

    if exc:
        raise exc

    return value


def image_grid(imgs: List[Image.Image]):
    "Make a grid of images"

    landscape: bool = imgs[0].size[1] >= imgs[0].size[0]
    dim = get_grid_dimension(len(imgs))
    if landscape:
        cols, rows = dim
    else:
        rows, cols = dim

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def convert_images_to_base64_grid(images: List[Image.Image]) -> str:
    "Convert a list of images to a list of base64 strings"

    return convert_image_to_base64(image_grid(images))


def resize(image: Image.Image, w: int, h: int):
    "Preprocess an image for the img2img procedure"

    return image.resize((w, h), resample=Image.LANCZOS)


def convert_bytes_to_image_stream(data: bytes) -> str:
    "Convert a base64 string to a PIL Image"

    pattern = re.compile(r"data:image\/[\w]+;base64,")

    img = data
    img = img.decode("utf-8")
    img = re.sub(pattern, "", img)

    return img
