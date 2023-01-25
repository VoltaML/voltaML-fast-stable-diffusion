# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import random
from typing import Callable, Dict, List, Optional, Union

import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.vae import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_safe.safety_checker import (
    CLIPVisionModel,
    SafeStableDiffusionSafetyChecker,
)
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.utils import deprecate, is_accelerate_available, logging
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from k_diffusion.sampling import get_sigmas_karras
from PIL import Image
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from core.functions import preprocess_image
from core.inference.unet_tracer import TracedUNet, get_traced_unet

logger = logging.get_logger(__name__)


class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        else:
            encoder_hidden_states = None

        assert encoder_hidden_states is not None
        return self.model(
            *args, encoder_hidden_states=encoder_hidden_states, **kwargs
        ).sample


class StableDiffusionKDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    <Tip warning={true}>

        This is an experimental pipeline and is likely to change in the future.

    </Tip>

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        sampler_noises=None,
    ):
        super().__init__()

        # get correct sigmas from LMS
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.vae: AutoencoderKL = self.vae
        self.text_encoder: CLIPTextModel = self.text_encoder
        self.unet: Union[UNet2DConditionModel, TracedUNet] = self.unet
        self.tokenizer: CLIPTokenizer = self.tokenizer
        self.scheduler: LMSDiscreteScheduler = self.scheduler
        self.safety_checker: Optional[
            SafeStableDiffusionSafetyChecker
        ] = self.safety_checker
        self.feature_extractor: Optional[CLIPVisionModel] = self.feature_extractor

        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        self.sampler_noises = sampler_noises

        model = ModelWrapper(unet, self.scheduler.alphas_cumprod)
        if scheduler.prediction_type == "v_prediction":  # type: ignore
            self.k_diffusion_model = CompVisVDenoiser(model)
        else:
            self.k_diffusion_model = CompVisDenoiser(model)

    def set_scheduler(self, scheduler_type: str):
        "Set the scheduler to use for the pipeline. Uses k_diffusion library."

        library = importlib.import_module("k_diffusion")
        sampling = getattr(library, "sampling")
        self.sampler = getattr(sampling, scheduler_type)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    def enable_traced_unet(self, model_id: str):
        "Loads a precomputed JIT traced U-Net model."

        traced_unet = get_traced_unet(model_id=model_id, pipe=self)
        if traced_unet is not None:
            self.unet = traced_unet
        else:
            raise ValueError(f"Traced U-Net model with id {model_id} does not exist.")

    def channel_last_memory(self):
        "Enable alternative way of ordering NCHW tensors"

        self.unet = self.unet.to(memory_format=torch.channels_last)  # type: ignore

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None  # type: ignore
            ):
                return torch.device(module._hf_hook.execution_device)  # type: ignore
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape  # type: ignore
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            if self.feature_extractor is not None:
                safety_checker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="pt"
                ).to(device)
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            has_nsfw_concept = None
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample  # type: ignore
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs_txt2img(self, prompt, height, width):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

    def prepare_latents_txt2img(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(
                    shape, generator=generator, device="cpu", dtype=dtype
                ).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    @torch.no_grad()
    def txt2img(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        input_latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[Dict], None]] = None,
        use_karras_sigmas: bool = True,
        seed: Optional[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # Apply torch seed
        if seed:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(random.randint(0, 100000000))

        # 0. Default height and width to unet
        sample_size = self.unet.config.sample_size  # type: ignore

        height = height or sample_size * self.vae_scale_factor
        width = width or sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs_txt2img(prompt, height, width)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True
        if guidance_scale <= 1.0:
            raise ValueError("has to use guidance_scale")

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=text_embeddings.device)

        logger.debug(f"Current sigmas: {self.scheduler.sigmas}")

        # 4.1. Prepare sigmas (or karras sigmas)
        if use_karras_sigmas:
            logger.debug("Applying KARRAS SIGMAS")
            kdiff_sigmas = self.k_diffusion_model.sigmas
            assert isinstance(kdiff_sigmas, torch.Tensor)
            sigma_min: float = kdiff_sigmas[0].item()
            sigma_max: float = kdiff_sigmas[-1].item()
            sigmas = get_sigmas_karras(
                n=num_inference_steps, sigma_min=sigma_min, sigma_max=sigma_max
            )
            sigmas = sigmas.to(text_embeddings.dtype).to(self.device)
            logger.debug("Applied Karras sigmas: %s", sigmas)
        else:
            sigmas = self.scheduler.sigmas
            sigmas = sigmas.to(text_embeddings.dtype)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels  # type: ignore
        latents = self.prepare_latents_txt2img(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=text_embeddings.dtype,
            device=device,
            generator=generator,
            latents=input_latents,
        )
        latents = latents * sigmas[0]
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        # 6. Define model function
        def model_fn(x: torch.Tensor, t: torch.Tensor):
            latent_model_input = torch.cat([x] * 2)
            t = torch.cat([t] * 2)

            noise_pred = self.k_diffusion_model(
                latent_model_input, t, cond=text_embeddings
            )

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            return noise_pred

        # 7. Run k-diffusion solver
        latents = self.sampler(
            model=model_fn,
            x=latents,
            sigmas=sigmas,
            callback=callback,
        )

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]],
        seed: Optional[int] = None,
        init_image: Optional[Union[torch.FloatTensor, Image.Image]] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[Dict], None]] = None,
        use_karras_sigmas: bool = True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if not init_image:
            raise ValueError(
                "You must provide an image to start the generation process."
            )

        # Apply torch seed
        if seed:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(random.randint(0, 100000000))

        # 1. Check inputs
        self.check_inputs_img2img(prompt, strength)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        assert guidance_scale is not None
        assert guidance_scale >= 1.0
        do_classifier_free_guidance = True

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Preprocess image
        image = preprocess_image(image=init_image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents_img2img(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # # 8. Denoising loop

        if use_karras_sigmas:
            logger.debug("Applying KARRAS SIGMAS")
            kdiff_sigmas = self.k_diffusion_model.sigmas
            assert isinstance(kdiff_sigmas, torch.Tensor)
            sigma_min: float = kdiff_sigmas[0].item()
            sigma_max: float = kdiff_sigmas[-1].item()
            sigmas = get_sigmas_karras(
                n=num_inference_steps, sigma_min=sigma_min, sigma_max=sigma_max
            )
            sigmas = sigmas.to(text_embeddings.dtype).to(self.device)
            logger.debug("Applied Karras sigmas: %s", sigmas)
        else:
            sigmas = self.scheduler.sigmas
            sigmas = sigmas.to(text_embeddings.dtype)

        # 5. Prepare latent variables
        latents = self.prepare_latents_img2img(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
        )
        latents = latents * sigmas[0]
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        # 6. Define model function
        def model_fn(x: torch.Tensor, t: torch.Tensor):
            latent_model_input = torch.cat([x] * 2)
            t = torch.cat([t] * 2)

            noise_pred = self.k_diffusion_model(
                latent_model_input, t, cond=text_embeddings
            )

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            return noise_pred

        # 7. Run k-diffusion solver
        latents = self.sampler(
            model=model_fn,
            x=latents,
            sigmas=sigmas,
            callback=callback,
        )

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def check_inputs_img2img(self, prompt, strength):
        "Check if all necessary arguments are valid."

        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [1.0, 1.0] but is {strength}"
            )

    def prepare_latents_img2img(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)  # type: ignore
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        rand_device = "cpu" if device.type == "mps" else device
        shape = init_latents.shape
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            noise = [
                torch.randn(
                    shape, generator=generator[i], device=rand_device, dtype=dtype
                )
                for i in range(batch_size)
            ]
            noise = torch.cat(noise, dim=0).to(device)
        else:
            noise = torch.randn(
                shape, generator=generator, device=rand_device, dtype=dtype
            ).to(device)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)  # type: ignore
        latents = init_latents

        return latents

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
