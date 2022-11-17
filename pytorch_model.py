from diffusers import StableDiffusionPipeline
import torch
from typing import List, Union
import time


def load_model(
    model_name_or_path="CompVis/stable-diffusion-v1-4"
) -> StableDiffusionPipeline:
    """Load model

    :param model_name_or_path: model name (downloaded from HF Hub) or model path (local), defaults to "CompVis/stable-diffusion-v1-4"
    :return: the Stable Diffusion pipeline
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path, 
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe = pipe.to("cuda")
    
    return pipe


def inference(
    model: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    img_height: int = 512,
    img_width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    seed: int = None,
    return_time=False
):
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
    if seed is not None:
        generator = torch.Generator(device='cuda')
        generator = generator.manual_seed(seed)

    start_time = time.time()
    with torch.autocast("cuda"):
        output = model(
            prompt=prompt,
            height=img_height,
            width=img_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        )
    end_time = time.time()
    
    if return_time:
        return output.images, end_time - start_time
    
    return output.images
    