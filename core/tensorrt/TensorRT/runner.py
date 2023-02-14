# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import gc
import json
import os
import random
import time
from typing import Optional

import diffusers
import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from tqdm import tqdm
from transformers import CLIPTokenizer

from ..runner import BaseRunner
from ..utilities import TRT_LOGGER, Engine
from .clip import create_clip_engine
from .models import CLIP, VAE, UNet
from .pwp import TensorRTPromptWeightingPipeline


def to_image(images):
    "Convert tensor into PIL images"

    images = (
        ((images + 1) * 255 / 2)
        .clamp(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .round()
        .type(torch.uint8)
        .cpu()
        .numpy()
    )
    result = []
    for i in range(images.shape[0]):
        result.append(Image.fromarray(images[i]))
    return result


def get_timesteps(
    scheduler,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
    is_text2img: bool,
):
    if is_text2img:
        return scheduler.timesteps.to(device), num_inference_steps
    else:
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:].to(device)
        return timesteps, num_inference_steps - t_start


def get_scheduler(scheduler_id: str):
    schedulers = {
        "ddim": diffusers.DDIMScheduler,
        "deis": diffusers.DEISMultistepScheduler,
        "dpm2": diffusers.KDPM2DiscreteScheduler,
        "dpm2-a": diffusers.KDPM2AncestralDiscreteScheduler,
        "euler_a": diffusers.EulerAncestralDiscreteScheduler,
        "euler": diffusers.EulerDiscreteScheduler,
        "heun": diffusers.DPMSolverMultistepScheduler,
        "dpm++": diffusers.DPMSolverMultistepScheduler,
        "dpm": diffusers.DPMSolverMultistepScheduler,
        "pndm": diffusers.PNDMScheduler,
    }
    return schedulers[scheduler_id]


def preprocess_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    "Preprocess the image to Tensor format"

    width, height = map(lambda x: x - x % 8, (width, height))
    image = image.resize(
        (width, height), resample=diffusers.utils.PIL_INTERPOLATION["lanczos"]  # type: ignore
    )
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array[None].transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_array)
    return 2.0 * image_tensor - 1.0


def prepare_latents(
    vae: VAE,
    vae_scale_factor: int,
    unet_in_channels: int,
    scheduler,
    image: Optional[Image.Image],
    timestep,
    batch_size,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
):
    if image is None:
        shape = (
            batch_size,
            unet_in_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents
    else:
        init_latent_dist = vae.encode(image).latent_dist  # type: ignore
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = torch.cat([0.18215 * init_latents] * batch_size, dim=0)
        shape = init_latents.shape
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = scheduler.add_noise(init_latents, noise, timestep)
        return latents


class TensorRTDiffusionRunner(BaseRunner):
    "Diffusion runner using TensorRT"

    def __init__(self, model_dir: str):
        meta_filepath = os.path.join(model_dir, "model_index.json")
        if not os.path.exists(meta_filepath):
            raise RuntimeError("Model meta data not found.")
        engine_dir = os.path.join(model_dir, "engine")

        with open(meta_filepath, mode="r", encoding="utf-8") as f:
            txt = f.read()
            self.meta = json.loads(txt)

        self.scheduler = None
        self.scheduler_id = None
        self.model_id = (
            self.meta["model_id"]
            if "model_id" in self.meta
            else "CompVis/stable-diffusion-v1-4"
        )
        self.device = "cuda"
        self.fp16 = self.meta["denoising_prec"] == "fp16"
        self.engines = {
            "clip": create_clip_engine(),
            "unet": Engine("unet", engine_dir),
            "vae": Engine("vae", engine_dir),
        }
        self.models = {
            "clip": CLIP(
                "openai/clip-vit-large-patch14", fp16=self.fp16, device=self.device
            ),
            "unet": UNet(
                self.meta["models"]["unet"], fp16=self.fp16, device=self.device
            ),
            "vae": VAE(self.meta["models"]["vae"], fp16=self.fp16, device=self.device),
        }
        self.en_vae = self.models["vae"].get_model()

        self.stream: cuda.Stream
        self.tokenizer: CLIPTokenizer
        self.pwp: TensorRTPromptWeightingPipeline
        self.loading: bool

    def activate(
        self,
        tokenizer_id="openai/clip-vit-large-patch14",
    ):
        "Load the model into memory"

        self.stream = cuda.Stream()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id)
        self.pwp = TensorRTPromptWeightingPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.engines["clip"],
            stream=self.stream,
            device=torch.device(self.device),
        )

        for engine in self.engines.values():
            engine.activate()
        self.loading = False

    def teardown(self):
        "Unload the model from memory"

        for engine in self.engines.values():
            del engine
        self.stream.free()
        del self.stream
        del self.tokenizer
        del self.pwp.text_encoder
        torch.cuda.empty_cache()
        gc.collect()

    def runEngine(self, model_name, feed_dict):
        "Run the engine"

        engine = self.engines[model_name]
        return engine.infer(feed_dict, self.stream)

    def infer(
        self,
        prompt: str,
        img: Optional[Image.Image] = None,
        negative_prompt: str = "",
        batch_size=1,
        batch_count=1,
        scheduler_id: str = "euler_a",
        steps=28,
        scale=28,
        image_height=512,
        image_width=512,
        seed=None,
        strength: float = 0.6,
    ):
        "Run the inference"

        self.wait_loading()

        if self.scheduler_id != scheduler_id:
            Scheduler = get_scheduler(scheduler_id)
            try:
                self.scheduler = Scheduler.from_pretrained(
                    self.model_id, subfolder="scheduler"
                )
            except Exception:  # pylint: disable=broad-except
                self.scheduler = Scheduler.from_config(
                    {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                    }
                )

        self.scheduler.set_timesteps(steps, device=self.device)  # type: ignore
        timesteps, steps = get_timesteps(
            self.scheduler, steps, strength, torch.device(self.device), img is None
        )
        latent_timestep = timesteps[:1].repeat(batch_size * batch_count)

        e2e_tic = time.perf_counter()

        results = []

        if seed is None:
            seed = random.randrange(0, 4294967294, 1)

        clip_perf_time = 0
        vae_perf_time = 0
        denoise_perf_time = 0

        for i in range(batch_count):
            for model_name, obj in self.models.items():
                self.engines[model_name].allocate_buffers(
                    shape_dict=obj.get_shape_dict(
                        batch_size, image_height, image_width
                    ),
                    device=self.device,
                )

            events = {}
            for stage in ["clip", "denoise", "vae"]:
                for marker in ["start", "stop"]:
                    events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

            manual_seed = seed + i
            generator = torch.Generator(device="cuda").manual_seed(manual_seed)

            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(  # type: ignore # pylint: disable=no-member
                TRT_LOGGER
            ):
                cudart.cudaEventRecord(events["clip-start"], 0)

                text_embeddings = self.pwp(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=scale,
                    batch_size=batch_size,
                    max_embeddings_multiples=1,
                )
                if self.fp16:
                    text_embeddings = text_embeddings.to(dtype=torch.float16)

                cudart.cudaEventRecord(events["clip-stop"], 0)

                if img is not None:
                    img = preprocess_image(img, image_height, image_width).to(  # type: ignore
                        device=self.device
                    )

                latents = prepare_latents(
                    vae=self.en_vae,
                    vae_scale_factor=8,
                    unet_in_channels=4,
                    scheduler=self.scheduler,
                    image=img,
                    timestep=latent_timestep,
                    batch_size=batch_size,
                    height=image_height,
                    width=image_width,
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                )

                torch.cuda.synchronize()

                cudart.cudaEventRecord(events["denoise-start"], 0)
                for _, timestep in enumerate(tqdm(iterable=timesteps)):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(  # type: ignore
                        latent_model_input, timestep
                    )

                    if timestep.dtype != torch.float32:
                        timestep_float = timestep.float()
                    else:
                        timestep_float = timestep

                    sample_inp = cuda.DeviceView(
                        ptr=latent_model_input.data_ptr(),
                        shape=latent_model_input.shape,
                        dtype=np.float32,
                    )
                    timestep_inp = cuda.DeviceView(
                        ptr=timestep_float.data_ptr(),
                        shape=timestep_float.shape,
                        dtype=np.float32,
                    )
                    embeddings_inp = cuda.DeviceView(
                        ptr=text_embeddings.data_ptr(),
                        shape=text_embeddings.shape,
                        dtype=np.float16 if self.fp16 else np.float32,
                    )

                    noise_pred = self.runEngine(
                        "unet",
                        {
                            "sample": sample_inp,
                            "timestep": timestep_inp,
                            "encoder_hidden_states": embeddings_inp,
                        },
                    )["latent"]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if scheduler_id in ["deis", "dpm2", "heun", "dpm++", "dpm", "pndm"]:
                        latents = self.scheduler.step(  # type: ignore
                            model_output=noise_pred, timestep=timestep, sample=latents
                        ).prev_sample
                    else:
                        latents = self.scheduler.step(  # type: ignore
                            model_output=noise_pred,
                            timestep=timestep,
                            sample=latents,
                            generator=generator,
                        ).prev_sample

                latents = 1.0 / 0.18215 * latents
                cudart.cudaEventRecord(events["denoise-stop"], 0)

                cudart.cudaEventRecord(events["vae-start"], 0)
                sample_inp = cuda.DeviceView(
                    ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
                )
                images = self.runEngine("vae", {"latent": sample_inp})["images"]
                cudart.cudaEventRecord(events["vae-stop"], 0)
                torch.cuda.synchronize()

                clip_perf_time = cudart.cudaEventElapsedTime(
                    events["clip-start"], events["clip-stop"]
                )[1]
                denoise_perf_time = cudart.cudaEventElapsedTime(
                    events["denoise-start"], events["denoise-stop"]
                )[1]
                vae_perf_time = cudart.cudaEventElapsedTime(
                    events["vae-start"], events["vae-stop"]
                )[1]

                results.append(
                    (
                        to_image(images),
                        {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "steps": steps,
                            "scale": scale,
                            "seed": manual_seed,
                            "height": image_height,
                            "width": image_width,
                            "img2img": img is not None,
                        },
                        {
                            "clip": clip_perf_time,
                            "denoise": denoise_perf_time,
                            "vae": vae_perf_time,
                        },
                    )
                )

        e2e_toc = time.perf_counter()
        all_perf_time = e2e_toc - e2e_tic
        print(
            f"all: {all_perf_time}s, clip: {clip_perf_time/1000}s, denoise: {denoise_perf_time/1000}s, vae: {vae_perf_time/1000}s"
        )

        return results, all_perf_time
