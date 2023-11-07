from typing import Callable, Literal, Optional, Union
import logging

import PIL
import torch
from tqdm import tqdm
from diffusers import LMSDiscreteScheduler, SchedulerMixin, StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
)
from transformers.models.clip import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)

from core.config import config
from core.inference.utilities import (
    prepare_latents,
    preprocess_image,
    preprocess_mask,
    prepare_mask_latents,
    prepare_mask_and_masked_image,
    prepare_extra_step_kwargs,
    get_weighted_text_embeddings,
    get_timesteps,
    full_vae,
    numpy_to_pil,
    philox,
    Placebo,
)
from core.scheduling import KdiffusionSchedulerAdapter
from core.optimizations import (
    inference_context,
    ensure_correct_device,
    unload_all,
)

# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class StableDiffusionXLLongPromptWeightingPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        parent,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        aesthetic_score: bool,
        force_zeros: bool,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,  # type: ignore
        )
        self.__init__additional__()

        self.parent = parent
        self.aesthetic_score: bool = aesthetic_score
        self.force_zeros: bool = force_zeros
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.text_encoder_2: CLIPTextModelWithProjection
        self.tokenizer: CLIPTokenizer
        self.tokenizer_2: CLIPTokenizer
        self.unet: UNet2DConditionModel
        self.scheduler: LMSDiscreteScheduler

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
        return torch.device(config.api.device)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        negative_prompt,
        max_embeddings_multiples,
        seed,
        prompt_expansion_settings=None,
    ):
        if negative_prompt == "":
            negative_prompt = None

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
            logger.debug(f"Post textual prompt: {prompt}")

            if negative_prompt is not None:
                logger.debug(f"Post textual negative_prompt: {negative_prompt}")

            obj = Placebo()
            ensure_correct_device(text_encoder)
            setattr(obj, "text_encoder", text_encoder)
            setattr(obj, "tokenizer", tokenizer)
            setattr(obj, "loras", [])

            (
                text_embeddings,
                pooled_embeddings,
                uncond_embeddings,
                uncond_pooled_embeddings,
            ) = get_weighted_text_embeddings(
                pipe=obj,  # type: ignore
                prompt=prompt,
                uncond_prompt="" if negative_prompt is None and not self.force_zeros else negative_prompt,  # type: ignore
                max_embeddings_multiples=max_embeddings_multiples,
                seed=seed,
                prompt_expansion_settings=prompt_expansion_settings,
            )
            if negative_prompt is None and self.force_zeros:
                uncond_embeddings = torch.zeros_like(text_embeddings)
                uncond_pooled_embeddings = torch.zeros_like(pooled_embeddings)
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            bs_embed, seq_len, _ = uncond_embeddings.shape  # type: ignore
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
            uncond_embeddings = uncond_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            prompt_embeds_list.append(text_embeddings)
            uncond_embeds_list.append(uncond_embeddings)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        uncond_embeds = torch.concat(uncond_embeds_list, dim=-1)

        bs_embed = pooled_embeddings.shape[0]  # type: ignore
        pooled_embeddings = pooled_embeddings.repeat(1, num_images_per_prompt)  # type: ignore
        pooled_embeddings = pooled_embeddings.view(bs_embed * num_images_per_prompt, -1)

        bs_embed = uncond_pooled_embeddings.shape[0]  # type: ignore
        uncond_pooled_embeddings = uncond_pooled_embeddings.repeat(1, num_images_per_prompt)  # type: ignore
        uncond_pooled_embeddings = uncond_pooled_embeddings.view(
            bs_embed * num_images_per_prompt, -1
        )

        # Only the last one is necessary
        return prompt_embeds.to(device), uncond_embeds.to(device), pooled_embeddings.to(device), uncond_pooled_embeddings.to(device)  # type: ignore

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        dtype,
    ):
        if self.aesthetic_score:
            add_time_ids = list(
                original_size + crops_coords_top_left + (aesthetic_score,)
            )
            add_neg_time_ids = list(
                original_size + crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim  # type: ignore
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features  # type: ignore

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim  # type: ignore
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim  # type: ignore
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        generator: Union[torch.Generator, philox.PhiloxGenerator],
        seed: int,
        negative_prompt: Optional[str] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        prompt_expansion_settings=None,
    ):
        aesthetic_score = 6.0
        negative_aesthetic_score = 2.5

        if config.api.torch_compile:
            self.unet = torch.compile(
                self.unet,
                fullgraph=config.api.torch_compile_fullgraph,
                dynamic=config.api.torch_compile_dynamic,
                mode=config.api.torch_compile_mode,
            )  # type: ignore

        # 0. Default height and width to unet
        with inference_context(self.unet, self.vae, height, width) as context:
            self.unet = context.unet  # type: ignore
            self.vae = context.vae  # type: ignore

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            split_latents_into_two = (
                config.api.dont_merge_latents and do_classifier_free_guidance
            )

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
                seed,
                prompt_expansion_settings=prompt_expansion_settings,
            )
            dtype = prompt_embeds.dtype

            # 4. Preprocess image and mask
            if isinstance(image, PIL.Image.Image):  # type: ignore
                image = preprocess_image(image)
            if image is not None:
                image = image.to(device=device, dtype=dtype)
            if isinstance(mask_image, PIL.Image.Image):  # type: ignore
                mask_image = preprocess_mask(mask_image)
            if mask_image is not None:
                mask, masked_image, _ = prepare_mask_and_masked_image(
                    image, mask_image, height, width
                )
                mask, _ = prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size * num_images_per_prompt,  # type: ignore
                    height,
                    width,
                    dtype,
                    device,
                    do_classifier_free_guidance,
                    self.vae,
                    self.vae_scale_factor,
                    self.vae.config.scaling_factor,  # type: ignore
                    generator=generator,
                )
            else:
                mask = None

            # 5. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)  # type: ignore
            timesteps, num_inference_steps = get_timesteps(
                self.scheduler,
                num_inference_steps,
                strength,
                device,
                image is None or hasattr(self, "controlnet"),
            )
            if isinstance(self.scheduler, KdiffusionSchedulerAdapter):
                self.scheduler.timesteps = timesteps
                self.scheduler.steps = num_inference_steps
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)  # type: ignore

            # 6. Prepare latent variables
            latents, init_latents_orig, noise = prepare_latents(
                self,  # type: ignore
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
            extra_step_kwargs = prepare_extra_step_kwargs(
                scheduler=self.scheduler, generator=generator, eta=eta
            )

            add_text_embeds = pooled_prompt_embeds
            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                (height, width),
                (0, 0),
                (height, width),
                aesthetic_score,
                negative_aesthetic_score,
                dtype,
            )

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)  # type: ignore

            # 8. Denoising loop
            def do_denoise(
                x: torch.Tensor,
                t: torch.IntTensor,
                call: Callable[..., torch.Tensor],
                change_source: Callable[[Callable], None],
            ) -> torch.Tensor:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([x] * 2) if do_classifier_free_guidance and not split_latents_into_two else x  # type: ignore
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

                if split_latents_into_two:
                    uncond, cond = prompt_embeds.chunk(2)
                    uncond_text, cond_text = add_text_embeds.chunk(2)
                    uncond_time, cond_time = add_time_ids.chunk(2)

                    added_cond_kwargs = {
                        "text_embeds": cond_text,
                        "time_ids": cond_time,
                    }
                    noise_pred_text = call(
                        latent_model_input,
                        t,
                        cond=cond,
                        added_cond_kwargs=added_cond_kwargs,
                    )

                    added_cond_kwargs = {
                        "text_embeds": uncond_text,
                        "time_ids": uncond_time,
                    }
                    noise_pred_uncond = call(
                        latent_model_input,
                        t,
                        cond=uncond,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                else:
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    noise_pred = call(  # type: ignore
                        latent_model_input,
                        t,
                        cond=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    if not split_latents_into_two:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # type: ignore
                    noise_pred = noise_pred_uncond + guidance_scale * (  # type: ignore
                        noise_pred_text - noise_pred_uncond  # type: ignore
                    )  # type: ignore

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, KdiffusionSchedulerAdapter):
                    # compute the previous noisy sample x_t -> x_t-1
                    x = self.scheduler.step(  # type: ignore
                        noise_pred, t.to(noise_pred.device), x.to(noise_pred.device), **extra_step_kwargs  # type: ignore
                    ).prev_sample  # type: ignore
                else:
                    x = noise_pred  # type: ignore

                if mask is not None:
                    # masking
                    init_latents_proper = self.scheduler.add_noise(  # type: ignore
                        init_latents_orig, noise, torch.tensor([t])  # type: ignore
                    )
                    x = (init_latents_proper * mask) + (x * (1 - mask))  # type: ignore
                return x

            ensure_correct_device(self.unet)
            if isinstance(self.scheduler, KdiffusionSchedulerAdapter):
                latents = self.scheduler.do_inference(
                    latents,  # type: ignore
                    generator=generator,
                    call=self.unet,  # type: ignore
                    apply_model=do_denoise,
                    callback=callback,
                    callback_steps=callback_steps,
                )
            else:
                s = self.unet

                def change(src):
                    nonlocal s
                    s = src

                def _call(*args, **kwargs):
                    if len(args) == 3:
                        encoder_hidden_states = args[-1]
                        args = args[:2]
                    if kwargs.get("cond", None) is not None:
                        encoder_hidden_states = kwargs.pop("cond")
                    return s(
                        *args,
                        encoder_hidden_states=encoder_hidden_states,  # type: ignore
                        return_dict=True,
                        **kwargs,
                    )[0]

                for i, t in enumerate(tqdm(timesteps, desc="SDXL")):
                    latents = do_denoise(latents, t, _call, change)  # type: ignore

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
            ensure_correct_device(self.vae)
            image = full_vae(latents, overwrite=lambda sample: self.vae.decode(sample).sample, height=height, width=width)  # type: ignore

            # 11. Convert to PIL
            if output_type == "pil":
                image = numpy_to_pil(image)

            unload_all()

            if not return_dict:
                return image, False

            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=False  # type: ignore
            )

    def text2img(
        self,
        generator: Union[torch.Generator, philox.PhiloxGenerator],
        prompt: str,
        seed: int,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        prompt_expansion_settings=None,
    ):
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
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
            prompt_expansion_settings=prompt_expansion_settings,
        )

    def img2img(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        prompt: str,
        seed: int,
        generator: Union[torch.Generator, philox.PhiloxGenerator],
        negative_prompt: Optional[str] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        prompt_expansion_settings=None,
    ):
        return self.__call__(
            prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,  # type: ignore
            guidance_scale=guidance_scale,  # type: ignore
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
            prompt_expansion_settings=prompt_expansion_settings,
        )

    def inpaint(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],  # type: ignore
        prompt: str,
        seed: int,
        generator: Union[torch.Generator, philox.PhiloxGenerator],
        negative_prompt: Optional[str] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        width: int = 512,
        height: int = 512,
        prompt_expansion_settings=None,
    ):
        return self.__call__(
            prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,  # type: ignore
            guidance_scale=guidance_scale,  # type: ignore
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
            prompt_expansion_settings=prompt_expansion_settings,
        )
