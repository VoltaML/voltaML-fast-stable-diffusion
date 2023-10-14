import numpy as np
import pytest
from diffusers.schedulers import KarrasDiffusionSchedulers
from PIL import Image

from core.inference.pytorch import PyTorchStableDiffusion
from core.types import (
    ControlNetData,
    ControlNetQueueEntry,
    Img2imgData,
    Img2ImgQueueEntry,
    InpaintData,
    InpaintQueueEntry,
    Txt2imgData,
    Txt2ImgQueueEntry,
)
from core.utils import convert_image_to_base64, unwrap_enum
from tests.const import KDIFF_SAMPLERS
from tests.functions import generate_random_image, generate_random_image_base64


@pytest.fixture(name="pipe")
def pipe_fixture():
    "Preloaded pipe that will be shared across all tests"

    return PyTorchStableDiffusion("Azher/Anything-v4.5-vae-fp16-diffuser")


@pytest.mark.parametrize("scheduler", list(KarrasDiffusionSchedulers) + KDIFF_SAMPLERS)
def test_txt2img_scheduler_sweep(
    pipe: PyTorchStableDiffusion, scheduler: KarrasDiffusionSchedulers
):
    "Sweep all schedulers with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=str(unwrap_enum(scheduler)),
            id="test",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


@pytest.mark.parametrize("height", [256, 512, 1024])
@pytest.mark.parametrize("width", [256, 512, 1024])
def test_txt2img_res_sweep(pipe: PyTorchStableDiffusion, height: int, width: int):
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


def test_txt2img_multi(pipe: PyTorchStableDiffusion):
    "Generating multiple images with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler,
            id="test",
            batch_size=2,
            batch_count=2,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    assert len(pipe.generate(job)) == 4


def test_txt2img_self_attention(pipe: PyTorchStableDiffusion):
    "Generate an image with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.DPMSolverMultistepScheduler,
            id="test",
            self_attention_scale=1,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_txt2img_karras_sigmas_diffusers(pipe: PyTorchStableDiffusion):
    "Generate an image with Text to Image"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.KDPM2AncestralDiscreteScheduler,
            id="test",
            sigmas="karras",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_txt2img_hr_fix(pipe: PyTorchStableDiffusion):
    "Generate an image with high resolution latent upscale"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
        flags={
            "high_resolution": {
                "scale": 2,
                "latent_scale_mode": "bilinear",
                "strength": 0.6,
                "steps": 50,
                "antialiased": False,
            }
        },
    )

    pipe.generate(job)


def test_img2img(pipe: PyTorchStableDiffusion):
    "Generate an image with Image to Image"

    job = Img2ImgQueueEntry(
        data=Img2imgData(
            image=generate_random_image_base64(),
            prompt="This is a test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_inpaint(pipe: PyTorchStableDiffusion):
    "Generate an image with Inpainting"

    np_mask = np.random.randint(0, 1, size=(256, 256, 3), dtype=np.uint8)
    mask = Image.fromarray(np_mask)
    encoded_mask = convert_image_to_base64(mask, prefix_js=False)

    job = InpaintQueueEntry(
        data=InpaintData(
            image=generate_random_image_base64(),
            prompt="This is a test",
            mask_image=encoded_mask,
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_controlnet(pipe: PyTorchStableDiffusion):
    "Generate an image with ControlNet Image to Image"

    job = ControlNetQueueEntry(
        data=ControlNetData(
            image=generate_random_image_base64(),
            prompt="This is a test",
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet="lllyasviel/sd-controlnet-canny",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_controlnet_preprocessed(pipe: PyTorchStableDiffusion):
    "Generate an image with ControlNet Image to Image while having the image preprocessed"

    from core.inference.utilities import image_to_controlnet_input

    preprocessed_image = generate_random_image()
    preprocessed_image = image_to_controlnet_input(
        preprocessed_image,
        data=ControlNetData(
            prompt="This is a test",
            id="test",
            image="",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet="lllyasviel/control_v11p_sd15_canny",
        ),
    )
    preprocessed_image_str = convert_image_to_base64(
        preprocessed_image, prefix_js=False
    )

    job = ControlNetQueueEntry(
        data=ControlNetData(
            image=preprocessed_image_str,
            prompt="This is a test",
            id="test",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            controlnet="lllyasviel/control_v11p_sd15_canny",
            is_preprocessed=True,
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_txt2img_with_lora(pipe: PyTorchStableDiffusion):
    "Generate an image with LoRA model"

    job = Txt2ImgQueueEntry(
        data=Txt2imgData(
            prompt="1girl, blonde, <lora:more_details:0.5>",
            scheduler=KarrasDiffusionSchedulers.UniPCMultistepScheduler,
            id="test",
        ),
        model="Azher/Anything-v4.5-vae-fp16-diffuser",
    )

    pipe.generate(job)


def test_unload(pipe: PyTorchStableDiffusion):
    "Unload the pipe from memory"

    pipe.unload()


def test_cleanup(pipe: PyTorchStableDiffusion):
    "Cleanup the memory"

    pipe.memory_cleanup()
