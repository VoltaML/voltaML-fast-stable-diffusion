import time
from typing import List, Optional, Union, Tuple

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as SDPipeType,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)
from PIL.Image import Image


def load_model(
    model_name_or_path="runwayml/stable-diffusion-v1-5",
    hf_token="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",
) -> SDPipeType:
    """Load model

    :param model_name_or_path: model name (downloaded from HF Hub) or model path (local), defaults to "runwayml/stable-diffusion-v1-5"
    :return: the Stable Diffusion pipeline
    """
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=hf_token,
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            use_auth_token=hf_token,
        )

    pipe = pipe.to("cuda")  # type: ignore - .to method is not typed

    return pipe


def inference(
    model: SDPipeType,
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]],
    img_height: int = 512,
    img_width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    seed: Optional[int] = None,
) -> Tuple[List[Image], float]:
    """Do inference

    :param model: the Stable Diffusion pipeline
    :param prompt: the prompt
    :param img_height: height of the generated image, defaults to 512
    :param img_width: width of the generated image, defaults to 512
    :param num_inference_steps: the number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference, defaults to 50
    :param guidance_scale: guidance scale, defaults to 7.5
    :param num_images_per_prompt: the number of images to generate per prompt, defaults to 1
    :param seed: Seed to make generation deterministic, defaults to None
    :param return_time: specify if time taken to generate the images should be returned, defaults to False
    :return: the output images and the time (if return time is True)
    """
    generator = None
    if seed:
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)

    start_time = time.time()

    with torch.autocast("cuda"):  # type: ignore - torch.autocast is not typed
        output = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=img_height,
            width=img_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )
    end_time = time.time()

    if isinstance(output, StableDiffusionPipelineOutput) and isinstance(
        output.images, list
    ):
        return (output.images, end_time - start_time)
    else:
        raise ValueError("Output is not of type StableDiffusionPipelineOutput")
