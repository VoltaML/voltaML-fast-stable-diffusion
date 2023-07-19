#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import inspect
import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from aitemplate.compiler import Model
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION
from transformers.models.clip import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image

from core.aitemplate.config import get_unet_in_channels
from core.aitemplate.src.modeling import mapping
from core.functions import init_ait_module

logger = logging.getLogger(__name__)


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class StableDiffusionAITPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for everything
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scheduler: KarrasDiffusionSchedulers,
        controlnet: Optional[ControlNetModel],
        unet: Optional[UNet2DConditionModel],  # type: ignore # pylint: disable=unused-argument
        directory: str = "",
        clip_ait_exe: Optional[Model] = None,
        unet_ait_exe: Optional[Model] = None,
        vae_ait_exe: Optional[Model] = None,
    ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        self.unet = unet

        self.controlnet = controlnet

        self.safety_checker: StableDiffusionSafetyChecker
        self.requires_safety_checker: bool
        self.feature_extractor: CLIPFeatureExtractor
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer

        self.batch: int = 1

        self.scheduler: LMSDiscreteScheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # type: ignore

        logger.debug(f"AIT workdir: {directory}")

        if clip_ait_exe is None:
            self.clip_ait_exe = init_ait_module(
                model_name="CLIPTextModel", workdir=directory
            )
            self.clip_ait_exe.set_many_constants_with_tensors(
                mapping.map_clip(self.text_encoder)
            )
            self.clip_ait_exe.fold_constants()
        else:
            self.clip_ait_exe = clip_ait_exe

        if unet_ait_exe is None:
            self.unet_ait_exe = init_ait_module(
                model_name="UNet2DConditionModel", workdir=directory
            )
            self.unet_ait_exe.set_many_constants_with_tensors(
                mapping.map_unet(self.unet)
            )
            self.unet_ait_exe.fold_constants()
        else:
            self.unet_ait_exe = unet_ait_exe

        if vae_ait_exe is None:
            self.vae_ait_exe = init_ait_module(
                model_name="AutoencoderKL", workdir=directory
            )
            self.vae_ait_exe.set_many_constants_with_tensors(
                mapping.map_vae(self.vae, encoder=False)
            )
            self.vae_ait_exe.fold_constants()
        else:
            self.vae_ait_exe = vae_ait_exe

        self.unet_in_channels = get_unet_in_channels(directory=Path(directory))

    def unet_inference(
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        height,
        width,
        down_block: list = [None],
        mid_block=None,
    ):
        exe_module = self.unet_ait_exe
        timesteps_pt = timesteps.expand(latent_model_input.shape[0])
        inputs = {
            "input0": latent_model_input.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        for i in range(12):
            if down_block[0] is not None:
                inputs[f"down_block_residual_{i}"] = (
                    down_block[i].permute((0, 2, 3, 1)).contiguous().cuda().half()
                )
        if mid_block is not None:
            inputs["mid_block_residual"] = (
                mid_block.permute((0, 2, 3, 1)).contiguous().cuda()
            )
        ys = []
        num_ouputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_ouputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch * 2
            shape[1] = height // 8
            shape[2] = width // 8
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=77):
        exe_module = self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input, height, width):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            shape[1] = height
            shape[2] = width
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out

    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [
                    np.array(
                        i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    )[None, :]
                    for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)  # type: ignore

        image_batch_size = image.shape[0]  # type: ignore

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)  # type: ignore

        image = image.to(device=device, dtype=dtype)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Image.Image] = None,  # type: ignore
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Optional[float] = 1.0,
        strength: Optional[float] = 0.7,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        num_images_per_prompt: int = 1,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        **kwargs,
    ):
        self.scheduler: LMSDiscreteScheduler

        if prompt is not None:
            if num_images_per_prompt == 1:
                assert isinstance(
                    prompt, str
                ), "When `num_images_per_prompt` is 1, `prompt` has to be of type `str`."
            else:
                if isinstance(prompt, str):
                    prompt = [prompt] * num_images_per_prompt
                elif isinstance(prompt, list):
                    assert (
                        len(prompt) == num_images_per_prompt
                    ), "When `num_images_per_prompt` is > 1, `prompt` has to be a list of length `num_images_per_prompt`."

                if negative_prompt is not None:
                    if isinstance(negative_prompt, str):
                        negative_prompt = [negative_prompt] * num_images_per_prompt
                    elif isinstance(negative_prompt, list):
                        assert (
                            len(negative_prompt) == num_images_per_prompt
                        ), "When `num_images_per_prompt` is > 1, `negative_prompt` has to be a list of length `num_images_per_prompt`."

            if isinstance(prompt, str):
                num_images_per_prompt = 1
            elif isinstance(prompt, list):
                num_images_per_prompt = len(prompt)
            else:
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        self.batch = num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        if prompt is not None:
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            prompt_embeds = self.clip_inference(text_input.input_ids.to(self.device))

            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                max_length = text_input.input_ids.shape[-1]
                if negative_prompt is None:
                    uncond_tokens = [""] * num_images_per_prompt
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif num_images_per_prompt != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {num_images_per_prompt}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = self.clip_inference(
                    uncond_input.input_ids.to(self.device)
                )

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])  # type: ignore
        else:
            text_embeddings = prompt_embeds

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        img2img = False
        self.scheduler.set_timesteps(num_inference_steps, **extra_step_kwargs)
        if image is not None and self.controlnet is None:
            if isinstance(image, Image.Image):
                init_image = preprocess(image)  # type: ignore

            img2img = True

            # convert to correct dtype and push to right device
            init_image = init_image.to(self.device, dtype=self.vae.dtype)  # type: ignore
            # encode the init image into latents and scale the latents
            init_latent_dist = self.vae.encode(init_image).latent_dist  # type: ignore
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for num_images_per_prompt
            init_latents = torch.cat([init_latents] * num_images_per_prompt)

            noise = torch.randn(
                init_latents.shape, generator=generator, device=self.device
            )

            # get the original timestep using init_timestep
            init_timestep = int(num_inference_steps * strength)  # type: ignore
            init_timestep = min(init_timestep, num_inference_steps)
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                timesteps = torch.tensor(
                    [num_inference_steps - init_timestep] * num_images_per_prompt,
                    device=self.device,
                ).to(dtype=torch.long)
            else:
                timesteps = self.scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor(
                    [timesteps] * num_images_per_prompt, device=self.device
                ).to(dtype=torch.long)
            latents = self.scheduler.add_noise(init_latents, noise, timesteps).to(  # type: ignore
                device=self.device
            )
        else:
            latents_device = "cpu" if self.device.type == "mps" else self.device
            latents_shape = (
                num_images_per_prompt,
                self.unet_in_channels,
                height // 8,
                width // 8,
            )
            if latents is None:
                latents = torch.randn(  # type: ignore
                    latents_shape,  # type: ignore
                    generator=generator,
                    device=latents_device,
                )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(
                        f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                    )
            latents = latents.to(self.device)  # type: ignore

            latents = latents * self.scheduler.init_noise_sigma  # type: ignore

        if image is not None and self.controlnet is not None:
            ctrl_image = self.prepare_image(
                image,
                width,
                height,
                self.batch,
                num_images_per_prompt,
                self.device,
                self.controlnet.dtype,
            )
            if do_classifier_free_guidance:
                ctrl_image = torch.cat([ctrl_image] * 2)

        text_embeddings = text_embeddings.half()  # type: ignore

        t_start = max(num_inference_steps - init_timestep, 0) if img2img else 0
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t_start + i
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).half()  # type: ignore

            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                latent_model_input = latent_model_input.to(self.vae.dtype)
                t = t.to(self.vae.dtype)

            # predict the noise residual
            if self.controlnet is not None and ctrl_image is not None:  # type: ignore
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=ctrl_image,  # type: ignore
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_pred = self.unet_inference(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    height=height,
                    width=width,
                    down_block=down_block_res_samples,
                    mid_block=mid_block_res_sample,
                )
            else:
                noise_pred = self.unet_inference(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    height=height,
                    width=width,
                )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, t_index, latents, **extra_step_kwargs
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

            if callback is not None:
                callback(i, t, latents)  # type: ignore

        latents = 1 / 0.18215 * latents  # type: ignore
        image: torch.Tensor = self.vae_inference(latents, height=height, width=width)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)  # type: ignore

            has_nsfw_concept = None
        elif output_type == "latent":
            image = latents  # type: ignore
            has_nsfw_concept = None
        else:
            raise ValueError(f"Invalid output_type {output_type}")

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept  # type: ignore
        )
