import pytest
from diffusers.schedulers import KarrasDiffusionSchedulers

from core.aitemplate.scripts.compile import compile_diffusers
from core.inference.aitemplate import AITemplateStableDiffusion
from core.types import Txt2imgData, Txt2ImgQueueEntry


@pytest.mark.slow
def test_compile_aitemplate_models():
    compile_diffusers(
        local_dir_or_id="andite/anything-v4.0",
    )


@pytest.fixture(name="pipe")
def pipe_fixture():
    return AITemplateStableDiffusion("andite--anything-v4.0__512x512x1")


def test_aitemplate_txt2img(pipe: AITemplateStableDiffusion):
    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="andite/anything-v4.0",
    )

    pipe.generate(job)


def test_unload(pipe: AITemplateStableDiffusion):
    pipe.unload()
