import base64
from io import BytesIO

from PIL.Image import Image


def convert_image_to_stream(image: Image) -> BytesIO:
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream

def convert_image_to_base64(image: Image) -> str:
    stream = convert_image_to_stream(image)
    return base64.b64encode(stream.read()).decode("utf-8")

def convert_base64_to_bytes(data: str):
    return BytesIO(base64.b64decode(data))