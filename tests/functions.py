import hashlib
import io

import numpy as np
from PIL import Image

from core.utils import convert_image_to_base64


def generate_random_image_base64(w: int = 512, h: int = 512) -> str:
    "Generate a random image and return it as a base64 encoded string"

    np_image = np.random.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    image = Image.fromarray(np_image)
    encoded_image = convert_image_to_base64(image, prefix_js=False)

    return encoded_image


def generate_random_image(w: int = 512, h: int = 512) -> Image.Image:
    "Generate a random image and return it as PIL Image"

    np_image = np.random.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    image = Image.fromarray(np_image)

    return image


def hash_image(image: Image.Image) -> str:
    "Return sha256 hash of image, computed partially so that it does not overflow memory"

    hash_ = hashlib.sha256()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    while True:
        data = image_bytes.read(65536)
        if not data:
            break
        hash_.update(data)

    image_bytes.close()
    return hash_.hexdigest()
