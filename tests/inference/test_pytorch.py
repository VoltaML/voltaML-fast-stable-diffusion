import pytest
from diffusers.schedulers import KarrasDiffusionSchedulers

from core.inference.pytorch import PyTorchStableDiffusion
from core.types import Txt2imgData, Txt2ImgQueueEntry


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


def test_unload(pipe: PyTorchStableDiffusion):
    pipe.unload()


def test_cleanup(pipe: PyTorchStableDiffusion):
    pipe.cleanup()
