import pytest
from diffusers.schedulers import KarrasDiffusionSchedulers

from core.types import (
    ControlNetData,
    ControlNetQueueEntry,
    Img2imgData,
    Img2ImgQueueEntry,
    Txt2imgData,
    Txt2ImgQueueEntry,
)
from tests.functions import generate_random_image_base64

try:
    from core.aitemplate.compile import compile_diffusers
    from core.inference.aitemplate import AITemplateStableDiffusion
except ModuleNotFoundError:
    pytest.skip("Skipping aitemplate tests, ait not installed", allow_module_level=True)


@pytest.mark.slow
def test_compile_aitemplate_models():
    compile_diffusers(
        local_dir_or_id="Azher--Anything-v4.5-vae-fp16-diffuser__512x512x1",
    )


@pytest.fixture(name="pipe")
def pipe_fixture():
    return AITemplateStableDiffusion(
        "Azher--Anything-v4.5-vae-fp16-diffuser__512x512x1"
    )


def test_aitemplate_txt2img(pipe: AITemplateStableDiffusion):
    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="Azher--Anything-v4.5-vae-fp16-diffuser__512x512x1",
    )

    pipe.generate(job)


def test_aitemplate_img2img(pipe: AITemplateStableDiffusion):
    job = Img2ImgQueueEntry(
        data=Img2imgData(
            prompt="test",
            image=generate_random_image_base64(),
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="Azher--Anything-v4.5-vae-fp16-diffuser__512x512x1",
    )

    pipe.generate(job)


def test_aitemplate_controlnet(pipe: AITemplateStableDiffusion):
    job = ControlNetQueueEntry(
        data=ControlNetData(
            prompt="test",
            image=generate_random_image_base64(),
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet="lllyasviel/sd-controlnet-canny",
            id="test",
        ),
        model="Azher--Anything-v4.5-vae-fp16-diffuser__512x512x1",
    )

    pipe.generate(job)


def test_unload(pipe: AITemplateStableDiffusion):
    pipe.unload()


def test_cleanup(pipe: AITemplateStableDiffusion):
    pipe.memory_cleanup()
