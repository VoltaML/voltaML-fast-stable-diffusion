import numpy as np
import pytest
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from PIL import Image

from core.inference.sdxl import SDXLStableDiffusion
from core.types import (
    Img2imgData,
    Img2ImgQueueEntry,
    InpaintData,
    InpaintQueueEntry,
    Txt2imgData,
    Txt2ImgQueueEntry,
)
from core.utils import convert_image_to_base64, unwrap_enum
from tests.const import KDIFF_SAMPLERS
from tests.functions import generate_random_image_base64


@pytest.fixture(name="pipe")
def pipe_fixture():
    "Preloaded pipe that will be shared across all tests"

    return SDXLStableDiffusion("data/models/sdxl")


@pytest.mark.parametrize("scheduler", list(KarrasDiffusionSchedulers) + KDIFF_SAMPLERS)
def test_txt2img_scheduler_sweep(
    pipe: SDXLStableDiffusion, scheduler: KarrasDiffusionSchedulers
):
    "Sweep all schedulers with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=str(unwrap_enum(scheduler)),
            id="test",
            width=128,
            height=128,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


@pytest.mark.parametrize("height", [128, 256, 512])
@pytest.mark.parametrize("width", [128, 256, 512])
def test_txt2img_res_sweep(pipe: SDXLStableDiffusion, height: int, width: int):
    "Sweep multiple resolutions with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler,
            id="test",
            height=height,
            width=width,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_txt2img_multi(pipe: SDXLStableDiffusion):
    "Generating multiple images with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler,
            id="test",
            batch_size=2,
            batch_count=2,
            width=128,
            height=128,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    assert len(pipe.generate(job)) == 4


def test_txt2img_self_attention(pipe: SDXLStableDiffusion):
    "Generate an image with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler,
            id="test",
            self_attention_scale=1,
            width=128,
            height=128,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_img2img(pipe: SDXLStableDiffusion):
    "Generate an image with Image to Image"

    job = Img2ImgQueueEntry(
        data=Img2imgData(
            image=generate_random_image_base64(),
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
            width=128,
            height=128,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_inpaint(pipe: SDXLStableDiffusion):
    "Generate an image with Inpainting"

    np_mask = np.random.randint(0, 1, size=(128, 128, 3), dtype=np.uint8)
    mask = Image.fromarray(np_mask)
    encoded_mask = convert_image_to_base64(mask, prefix_js=False)

    job = InpaintQueueEntry(
        data=InpaintData(
            image=generate_random_image_base64(),
            prompt="This is a test",
            mask_image=encoded_mask,
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            width=128,
            height=128,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_unload(pipe: SDXLStableDiffusion):
    "Unload the pipe from memory"

    pipe.unload()


def test_cleanup(pipe: SDXLStableDiffusion):
    "Cleanup the memory"

    pipe.memory_cleanup()
