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
from tests.functions import generate_random_image


@pytest.fixture(name="pipe")
def pipe_fixture():
    "Preloaded pipe that will be shared across all tests"

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
    job = Img2ImgQueueEntry(
        data=Img2imgData(
            image=generate_random_image(),
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_inpaint(pipe: PyTorchStableDiffusion):
    np_mask = np.random.randint(0, 1, size=(256, 256, 3), dtype=np.uint8)
    mask = Image.fromarray(np_mask)
    encoded_mask = convert_image_to_base64(mask)

    job = InpaintQueueEntry(
        data=InpaintData(
            image=generate_random_image(),
            prompt="This is a test",
            mask_image=encoded_mask,
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_control_net(pipe: PyTorchStableDiffusion):
    job = ControlNetQueueEntry(
        data=ControlNetData(
            image=generate_random_image(),
            prompt="This is a test",
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet=ControlNetMode.CANNY,
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_lora(pipe: PyTorchStableDiffusion):
    pipe.load_lora("data/lora/shenheLoraCollection_shenheHard.safetensors")


def test_txt2img_with_lora(pipe: PyTorchStableDiffusion):
    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="shenhe (genshin)",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_unload(pipe: PyTorchStableDiffusion):
    pipe.unload()


def test_cleanup(pipe: PyTorchStableDiffusion):
    pipe.memory_cleanup()
