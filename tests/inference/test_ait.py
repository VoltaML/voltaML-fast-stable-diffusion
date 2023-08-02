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
    from core.inference.ait import AITemplateStableDiffusion
except ModuleNotFoundError:
    pytest.skip("Skipping aitemplate tests, ait not installed", allow_module_level=True)

model = "Azher--Anything-v4.5-vae-fp16-diffuser__512-1024x512-1024x1-1"


@pytest.fixture(name="pipe")
def pipe_fixture():
    return AITemplateStableDiffusion(
        model_id=model,
    )


@pytest.mark.parametrize("scheduler", list(KarrasDiffusionSchedulers))
def test_aitemplate_txt2img(
    pipe: AITemplateStableDiffusion, scheduler: KarrasDiffusionSchedulers
):
    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=scheduler,
            id="test",
        ),
        model=model,
    )

    pipe.generate(job)


@pytest.mark.parametrize("scheduler", list(KarrasDiffusionSchedulers))
def test_aitemplate_img2img(
    pipe: AITemplateStableDiffusion, scheduler: KarrasDiffusionSchedulers
):
    job = Img2ImgQueueEntry(
        data=Img2imgData(
            prompt="test",
            image=generate_random_image_base64(),
            scheduler=scheduler,
            id="test",
        ),
        model=model,
    )

    pipe.generate(job)


@pytest.mark.parametrize("scheduler", list(KarrasDiffusionSchedulers))
def test_aitemplate_controlnet(
    pipe: AITemplateStableDiffusion, scheduler: KarrasDiffusionSchedulers
):
    job = ControlNetQueueEntry(
        data=ControlNetData(
            prompt="test",
            image=generate_random_image_base64(),
            scheduler=scheduler,
            controlnet="lllyasviel/sd-controlnet-canny",
            id="test",
        ),
        model=model,
    )

    pipe.generate(job)


def test_unload(pipe: AITemplateStableDiffusion):
    pipe.unload()


def test_cleanup(pipe: AITemplateStableDiffusion):
    pipe.memory_cleanup()
