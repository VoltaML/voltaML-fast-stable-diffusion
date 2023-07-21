# HuggingFace example pipeline taken from https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py

import inspect
from contextlib import ExitStack
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import LMSDiscreteScheduler, SchedulerMixin, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.utils import PIL_INTERPOLATION, logging
from transformers.models.clip import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from core.config import config
from core.inference.pytorch.latents import prepare_latents
from core.inference.pytorch.lwp import get_weighted_text_embeddings
from core.inference.pytorch.sag import (
    CrossAttnStoreProcessor,
    pred_epsilon,
    pred_x0,
    sag_masking,
)
from core.optimizations import autocast

# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask, scale_factor=8):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize(
        (w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"]
    )
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


class StableDiffusionLongPromptWeightingPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
    weighting in prompt.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

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
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        parent,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,  # type: ignore
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        self.__init__additional__()

        self.parent = parent
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.unet: UNet2DConditionModel
        self.scheduler: LMSDiscreteScheduler
        self.safety_checker: StableDiffusionSafetyChecker
        self.feature_extractor: CLIPFeatureExtractor
        self.requires_safety_checker: bool

    def __init__additional__(self):
        if not hasattr(self, "vae_scale_factor"):
            setattr(
                self,
                "vae_scale_factor",
                2 ** (len(self.vae.config.block_out_channels) - 1),  # type: ignore
            )

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):  # type: ignore
            return self.device
        for module in self.unet.modules():  # type: ignore
            if (
                hasattr(module, "_hf_hook")
                and hasattr(
                    module._hf_hook,  # pylint: disable=protected-access
                    "execution_device",
                )
                and module._hf_hook.execution_device  # pylint: disable=protected-access # type: ignore
                is not None
            ):
                return torch.device(
                    module._hf_hook.execution_device  # pylint: disable=protected-access # type: ignore
                )
        return self.device

    def upcast_vae(self):
        from diffusers.models.attention_processor import (
            XFormersAttnProcessor,
            AttnProcessor2_0,
        )

        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,  # type: ignore
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)  # type: ignore

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        negative_prompt,
        max_embeddings_multiples,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        prompts = [prompt, prompt]
        negative_prompts = [negative_prompt, negative_prompt]
        tokenizers = (
            [self.tokenizer, self.tokenizer_2]
            if self.tokenizer is not None
            else [self.tokenizer_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )

        prompt_embeds_list = []
        uncond_embeds_list = []
        for prompt, negative_prompt, tokenizer, text_encoder in zip(
            prompts, negative_prompts, tokenizers, text_encoders
        ):
            prompt = self.maybe_convert_prompt(prompt, tokenizer)
            logger.debug(f"Post textual prompt: {prompt}")

            if negative_prompt is not None:
                negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)
                logger.debug(f"Post textual negative_prompt: {negative_prompt}")

            obj = object()
            obj.text_encoder = text_encoder  # type: ignore
            obj.tokenizer = tokenizer  # type: ignore
            obj.loras = []  # type: ignore

            text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
                pipe=obj,  # type: ignore
                prompt=prompt,
                uncond_prompt=None if negative_prompt is None else [negative_prompt] * batch_size,  # type: ignore
                max_embeddings_multiples=max_embeddings_multiples,
            )
            pooled_embeddings = text_embeddings.hidden_states
            if negative_prompt is None:
                uncond_embeddings = torch.zeros_like(text_embeddings)
                uncond_pooled_embeddings = torch.zeros_like(pooled_embeddings)
            else:
                uncond_pooled_embeddings = uncond_embeddings.hidden_states  # type: ignore
            prompt_embeds_list.append(pooled_embeddings)
            uncond_embeds_list.append(uncond_pooled_embeddings)

        pooled_embeddings = torch.concat(prompt_embeds_list, dim=-1)
        uncond_pooled_embeddings = torch.concat(uncond_embeds_list, dim=-1)

        bs_embed = pooled_embeddings.shape[0]
        pooled_embeddings = pooled_embeddings.repeat(1, num_images_per_prompt)
        pooled_embeddings = pooled_embeddings.view(bs_embed * num_images_per_prompt, -1)

        bs_embed = uncond_pooled_embeddings.shape[0]  # type: ignore
        uncond_pooled_embeddings = uncond_pooled_embeddings.repeat(1, num_images_per_prompt)  # type: ignore
        uncond_pooled_embeddings = uncond_pooled_embeddings.view(
            bs_embed * num_images_per_prompt, -1
        )

        # Only the last one is necessary
        return pooled_embeddings.to(device), uncond_pooled_embeddings.to(device), text_embeddings.to(device), uncond_embeddings.to(device)  # type: ignore

    def check_inputs(self, prompt, height, width, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device, is_text2img):
        if is_text2img:
            return self.scheduler.timesteps.to(device), num_inference_steps  # type: ignore
        else:
            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)  # type: ignore
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(device)  # type: ignore
            return timesteps, num_inference_steps - t_start

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim  # type: ignore
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features  # type: ignore

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample  # type: ignore
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()  # type: ignore
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()  # type: ignore
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        self_attention_scale: float = 0.0,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        if config.api.torch_compile:
            self.unet = torch.compile(
                self.unet,
                fullgraph=config.api.torch_compile_fullgraph,
                dynamic=config.api.torch_compile_dynamic,
                mode=config.api.torch_compile_mode,
            )  # type: ignore

        # 0. Default height and width to unet
        with autocast(
            dtype=self.unet.dtype,
            disable=not config.api.autocast,
        ):
            height = height or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore
            width = width or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, strength, callback_steps)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            do_self_attention_guidance = self_attention_scale > 0.0

            # 3. Encode input prompt
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                negative_prompt,
                max_embeddings_multiples,
            )
            dtype = prompt_embeds.dtype

            # 4. Preprocess image and mask
            if isinstance(image, PIL.Image.Image):  # type: ignore
                image = preprocess_image(image)
            if image is not None:
                image = image.to(device=self.device, dtype=dtype)
            if isinstance(mask_image, PIL.Image.Image):  # type: ignore
                mask_image = preprocess_mask(mask_image, self.vae_scale_factor)
            if mask_image is not None:
                mask = mask_image.to(device=self.device, dtype=dtype)
                mask = torch.cat([mask] * batch_size * num_images_per_prompt)  # type: ignore
            else:
                mask = None

            # 5. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)  # type: ignore
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device, image is None
            )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)  # type: ignore

            # 6. Prepare latent variables
            latents, init_latents_orig, noise = prepare_latents(
                self,
                image,
                latent_timestep,
                batch_size * num_images_per_prompt,  # type: ignore
                height,
                width,
                dtype,
                device,
                generator,
                latents,
            )

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            if do_self_attention_guidance:
                store_processor = CrossAttnStoreProcessor()
                self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor  # type: ignore

            map_size = None

            def get_map_size(_, __, output):
                nonlocal map_size
                map_size = output[0].shape[
                    -2:
                ]  # output.sample.shape[-2:] in older diffusers

            # 8. Denoising loop
            with ExitStack() as gs:
                if do_self_attention_guidance:
                    gs.enter_context(self.unet.mid_block.attentions[0].register_forward_hook(get_map_size))  # type: ignore

                for i, t in enumerate(self.progress_bar(timesteps)):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # type: ignore
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

                    # predict the noise residual
                    noise_pred = self.unet(  # type: ignore
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if do_self_attention_guidance:
                        if do_classifier_free_guidance:
                            pred = pred_x0(self, latents, noise_pred_uncond, t)  # type: ignore
                            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)  # type: ignore
                            degraded_latents = sag_masking(
                                self, pred, uncond_attn, map_size, t, pred_epsilon(self, latents, noise_pred_uncond, t)  # type: ignore
                            )
                            uncond_emb, _ = text_embeddings.chunk(2)
                            # predict the noise residual
                            # this probably could have been done better but honestly fuck this
                            degraded_prep = self.unet(  # type: ignore
                                degraded_latents,
                                t,
                                encoder_hidden_states=uncond_emb,
                            ).sample
                            noise_pred += self_attention_scale * (noise_pred_uncond - degraded_prep)  # type: ignore
                        else:
                            pred = pred_x0(self, latents, noise_pred, t)
                            cond_attn = store_processor.attention_probs  # type: ignore
                            degraded_latents = sag_masking(
                                self,
                                pred,
                                cond_attn,
                                map_size,
                                t,
                                pred_epsilon(self, latents, noise_pred, t),
                            )
                            # predict the noise residual
                            degraded_prep = self.unet(  # type: ignore
                                degraded_latents,
                                t,
                                encoder_hidden_states=text_embeddings,
                            ).sample
                            noise_pred += self_attention_scale * (noise_pred - degraded_prep)  # type: ignore

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(  # type: ignore
                        noise_pred, t.to(noise_pred.device), latents.to(noise_pred.device), **extra_step_kwargs  # type: ignore
                    ).prev_sample  # type: ignore

                    if mask is not None:
                        # masking
                        init_latents_proper = self.scheduler.add_noise(  # type: ignore
                            init_latents_orig, noise, torch.tensor([t])  # type: ignore
                        )
                        latents = (init_latents_proper * mask) + (latents * (1 - mask))  # type: ignore

                    # call the callback, if provided
                    if i % callback_steps == 0:
                        if callback is not None:
                            callback(i, t, latents)  # type: ignore
                        if (
                            is_cancelled_callback is not None
                            and is_cancelled_callback()
                        ):
                            return None

            # 9. Post-processing
            if output_type == "latent":
                return latents, False

            image = self.decode_latents(latents)

            # 10. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, text_embeddings.dtype
            )

            # 11. Convert to PIL
            if output_type == "pil":
                image = self.numpy_to_pil(image)

            if hasattr(self, "final_offload_hook"):
                self.final_offload_hook.offload()  # type: ignore

            if not return_dict:
                return image, has_nsfw_concept

            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept  # type: ignore
            )

    def text2img(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        self_attention_scale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            self_attention_scale=self_attention_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )

    def img2img(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        self_attention_scale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,  # type: ignore
            guidance_scale=guidance_scale,  # type: ignore
            self_attention_scale=self_attention_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,  # type: ignore
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )

    def inpaint(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        self_attention_scale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        width: int = 512,
        height: int = 512,
    ):
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,  # type: ignore
            guidance_scale=guidance_scale,  # type: ignore
            self_attention_scale=self_attention_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,  # type: ignore
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
            width=width,
            height=height,
        )
