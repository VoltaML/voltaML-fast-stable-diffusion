import numpy as np
import pytest
from diffusers.schedulers import KarrasDiffusionSchedulers
from PIL import Image

from core.inference.pytorch import PyTorchStableDiffusion
from core.types import (
    ControlNetData,
    ControlNetMode,
    ControlNetQueueEntry,
    Img2imgData,
    Img2ImgQueueEntry,
    InpaintData,
    InpaintQueueEntry,
    Txt2imgData,
    Txt2ImgQueueEntry,
)
from core.utils import convert_image_to_base64


@pytest.fixture(name="pipe")
def pipe_fixture():
    return PyTorchStableDiffusion("andite/anything-v4.0")


def test_txt2img(pipe: PyTorchStableDiffusion):
    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_img2img(pipe: PyTorchStableDiffusion):
    np_image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(np_image)
    encoded_image = convert_image_to_base64(image)

    job = Img2ImgQueueEntry(
        data=Img2imgData(
            image=encoded_image,
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_inpaint(pipe: PyTorchStableDiffusion):
    np_image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(np_image)
    encoded_image = convert_image_to_base64(image)

    np_mask = np.random.randint(0, 1, size=(256, 256, 3), dtype=np.uint8)
    mask = Image.fromarray(np_mask)
    encoded_mask = convert_image_to_base64(mask)

    job = InpaintQueueEntry(
        data=InpaintData(
            image=encoded_image,
            prompt="This is a test",
            mask_image=encoded_mask,
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_control_net(pipe: PyTorchStableDiffusion):
    np_image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(np_image)
    encoded_image = convert_image_to_base64(image)

    job = ControlNetQueueEntry(
        data=ControlNetData(
            image=encoded_image,
            prompt="This is a test",
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet=ControlNetMode.CANNY,
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_unload(pipe: PyTorchStableDiffusion):
    pipe.unload()


def test_cleanup(pipe: PyTorchStableDiffusion):
    pipe.memory_cleanup()
