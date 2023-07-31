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
import logging
from pathlib import Path
from typing import Callable, List, Optional, Union
import math

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
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers.models.clip import CLIPTextModel, CLIPTokenizer
from PIL import Image

from core.aitemplate.config import get_unet_in_channels
from core.aitemplate.src.modeling import mapping
from core.inference.utilities import (
    get_weighted_text_embeddings,
    prepare_latents,
    get_timesteps,
    init_ait_module,
    prepare_image,
    preprocess_image,
    prepare_extra_step_kwargs,
    progress_bar,
)

logger = logging.getLogger(__name__)


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

    def unet_inference(  # pylint: disable=dangerous-default-value
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        height,
        width,
        down_block: list = [None],
        mid_block=None,
    ):
        "Execute AIT#UNet module"
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
            shape[1] = math.ceil(height / 8)
            shape[2] = math.ceil(width / 8)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=77):
        "Execute AIT#CLIP module"
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
        "Execute AIT#AutoencoderKL module"
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            shape[0] = self.batch
            shape[1] = math.ceil(height / 8) * 8
            shape[2] = math.ceil(width / 8) * 8
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out[:, :, :height, :width]

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
        guess_mode: Optional[bool] = False,
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

        self.batch = num_images_per_prompt

        if self.controlnet is not None:
            global_pool_conditions = self.controlnet.config.global_pool_conditions  # type: ignore
            guess_mode = guess_mode or global_pool_conditions

        do_classifier_free_guidance = guidance_scale > 1.0
        if prompt is not None:
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                self,
                prompt=prompt,
                uncond_prompt=negative_prompt,
                max_embeddings_multiples=3,
            )
        assert prompt_embeds is not None

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])  # type: ignore
        else:
            text_embeddings = prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)  # type: ignore
        txt2img = image is None or self.controlnet is not None
        timesteps, num_inference_steps = get_timesteps(
            self.scheduler, num_inference_steps, strength or 0.7, self.device, txt2img
        )
        latent_timestep = timesteps[:1].repeat(self.batch * num_images_per_prompt)  # type: ignore

        if image is not None:
            if isinstance(image, Image.Image):
                width, height = image.size  # type: ignore

            if self.controlnet is None:
                image = preprocess_image(image)
            else:
                ctrl_image = prepare_image(
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

        latents, _, _ = prepare_latents(
            self,
            image if self.controlnet is None else None,  # type: ignore
            latent_timestep,
            self.batch,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            align_to=64,
        )
        extra_step_kwargs = prepare_extra_step_kwargs(self.scheduler, generator, eta)  # type: ignore
        # Necessary for controlnet to function
        text_embeddings = text_embeddings.half()  # type: ignore

        controlnet_keep = []
        if self.controlnet is not None:
            for i in range(len(timesteps)):
                controlnet_keep.append(
                    1.0
                    - float(i / len(timesteps) < 0.0 or (i + 1) / len(timesteps) > 1.0)
                )

        for i, t in enumerate(progress_bar(timesteps)):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # type: ignore
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).half()  # type: ignore

            # predict the noise residual
            if self.controlnet is not None and ctrl_image is not None:  # type: ignore
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(
                        control_model_input, t
                    ).half()
                    controlnet_prompt_embeds = text_embeddings.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = text_embeddings

                cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds.half(),
                    controlnet_cond=ctrl_image,  # type: ignore
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [
                        torch.cat([torch.zeros_like(d), d])
                        for d in down_block_res_samples
                    ]
                    mid_block_res_sample = torch.cat(
                        [
                            torch.zeros_like(mid_block_res_sample),
                            mid_block_res_sample,
                        ]
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

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False  # type: ignore
            )[0]

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
