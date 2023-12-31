import logging
from contextlib import ExitStack
from typing import Callable, List, Literal, Optional, Union
import inspect

import PIL
import torch
from diffusers.models.adapter import MultiAdapter
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image
from tqdm import tqdm
from transformers.models.clip import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from core.config import config
from core.flags import SDXLRefinerFlag, DeepshrinkFlag, ScalecrafterFlag
from core.inference.utilities import (
    calculate_cfg,
    full_vae,
    get_timesteps,
    get_weighted_text_embeddings,
    numpy_to_pil,
    philox,
    prepare_extra_step_kwargs,
    prepare_latents,
    prepare_mask_and_masked_image,
    prepare_mask_latents,
    preprocess_adapter_image,
    preprocess_image,
    preprocess_mask,
    sag,
    modify_kohya,
    postprocess_kohya,
    get_scalecrafter_config,
    post_scalecrafter,
    step_scalecrafter,
    setup_scalecrafter,
    ScalecrafterSettings,
)
from core.optimizations import ensure_correct_device, inference_context, unload_all
from core.scheduling import KdiffusionSchedulerAdapter

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

    def _default_height_width(self, height, width, image):
        if image is None:
            return height, width

        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[-2]

            # round down to nearest multiple of `self.adapter.downscale_factor`
            if hasattr(self, "adapter") and self.adapter is not None:
                height = (
                    height // self.adapter.downscale_factor
                ) * self.adapter.downscale_factor

        if width is None:
            if isinstance(image, Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[-1]

            # round down to nearest multiple of `self.adapter.downscale_factor`
            if hasattr(self, "adapter") and self.adapter is not None:
                width = (
                    width // self.adapter.downscale_factor
                ) * self.adapter.downscale_factor

        return height, width

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
            ensure_correct_device(text_encoder)
            prompt = self.maybe_convert_prompt(prompt, tokenizer)
            logger.debug(f"Post textual prompt: {prompt}")

            if negative_prompt is not None:
                negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)
                logger.debug(f"Post textual negative_prompt: {negative_prompt}")

            (
                text_embeddings,
                pooled_embeddings,
                uncond_embeddings,
                uncond_pooled_embeddings,
            ) = get_weighted_text_embeddings(
                pipe=self.parent,  # type: ignore
                prompt=prompt,
                uncond_prompt="" if negative_prompt is None and not self.force_zeros else negative_prompt,  # type: ignore
                max_embeddings_multiples=max_embeddings_multiples,
                seed=seed,
                prompt_expansion_settings=prompt_expansion_settings,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
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
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        original_size: Optional[List[int]] = [1024, 1024],
        negative_prompt: Optional[str] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,  # type: ignore
        self_attention_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 100,
        output_type: Literal["pil", "latent"] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        prompt_expansion_settings=None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        adapter_conditioning_factor: float = 1.0,
        refiner: Optional[SDXLRefinerFlag] = None,
        refiner_model: Optional["StableDiffusionXLLongPromptWeightingPipeline"] = None,
        deepshrink: Optional[DeepshrinkFlag] = None,
        scalecrafter: Optional[ScalecrafterFlag] = None,  # type: ignore
    ):
        if original_size is None:
            original_size = [height, width]

        if config.api.torch_compile:
            self.unet = torch.compile(
                self.unet,
                fullgraph=config.api.torch_compile_fullgraph,
                dynamic=config.api.torch_compile_dynamic,
                mode=config.api.torch_compile_mode,
            )  # type: ignore

        # 0. Default height and width to unet
        with inference_context(self.unet, self.vae, height, width) as context:  # type: ignore
            self.unet = context.unet  # type: ignore
            self.vae = context.vae  # type: ignore

            if scalecrafter is not None:
                unsafe = scalecrafter.unsafe_resolutions  # type: ignore
                scalecrafter: ScalecrafterSettings = get_scalecrafter_config("sd15", height, width, scalecrafter.disperse)  # type: ignore
                logger.info(
                    f'Applying ScaleCrafter with (base="{scalecrafter.base}", res="{scalecrafter.height * 8}x{scalecrafter.width * 8}", dis="{scalecrafter.disperse is not None}")'
                )
                if not unsafe and (
                    (scalecrafter.height * 8) != height
                    or (scalecrafter.width * 8) != width
                ):
                    height, width = scalecrafter.height * 8, scalecrafter.width * 8

            refiner_steps = 10000
            if refiner_model is not None:
                assert refiner is not None
                num_inference_steps += refiner.steps
                refiner_steps = num_inference_steps // (refiner.strength + 1)

                refiner_model = refiner_model.unet  # type: ignore
                aesthetic_score = refiner.aesthetic_score
                negative_aesthetic_score = refiner.negative_aesthetic_score

            original_size = tuple(original_size)  # type: ignore
            height, width = self._default_height_width(height, width, image)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            do_self_attention_guidance = self_attention_scale > 1.0
            split_latents_into_two = (
                not config.api.batch_cond_uncond and do_classifier_free_guidance
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

            adapter_input = None  # type: ignore
            if hasattr(self, "adapter"):
                if isinstance(self.adapter, MultiAdapter):
                    adapter_input: list = []  # type: ignore

                    if not isinstance(adapter_conditioning_scale, list):
                        adapter_conditioning_scale = [
                            adapter_conditioning_scale * len(image)
                        ]

                    for oi in image:
                        oi = preprocess_adapter_image(oi, height, width)
                        oi = oi.to(device, dtype)  # type: ignore
                        adapter_input.append(oi)  # type: ignore
                else:
                    adapter_input: torch.Tensor = preprocess_adapter_image(  # type: ignore
                        adapter_input, height, width
                    )
                    adapter_input.to(device, dtype)

            # 4. Preprocess image and mask
            if isinstance(image, Image.Image):
                image = preprocess_image(image)
            if image is not None:
                image = image.to(device=device, dtype=dtype)
            if isinstance(mask_image, Image.Image):
                mask_image = preprocess_mask(mask_image)
                mask_image = mask_image.to(device=device, dtype=dtype)
            if mask_image is not None:
                mask, masked_image, _ = prepare_mask_and_masked_image(
                    image, mask_image, height, width
                )
                mask, _ = prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size * num_images_per_prompt,
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
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            # 6. Prepare latent variables
            latents, init_latents_orig, noise = prepare_latents(
                self,  # type: ignore
                image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                height,
                width,
                dtype,
                device,
                generator,
                None,
                latents,
            )

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = prepare_extra_step_kwargs(
                scheduler=self.scheduler, generator=generator, eta=eta, device=device
            )

            setup_scalecrafter(self.unet, scalecrafter)  # type: ignore

            if hasattr(self, "adapter"):
                if isinstance(self.adapter, MultiAdapter):
                    adapter_state = self.adapter(
                        adapter_input, adapter_conditioning_scale
                    )
                    for k, v in enumerate(adapter_state):
                        adapter_state[k] = v
                else:
                    adapter_state = self.adapter(adapter_input)
                    for k, v in enumerate(adapter_state):
                        adapter_state[k] = v * adapter_conditioning_scale
                if num_images_per_prompt > 1:
                    for k, v in enumerate(adapter_state):
                        adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
                if do_classifier_free_guidance:
                    for k, v in enumerate(adapter_state):
                        adapter_state[k] = torch.cat([v] * 2, dim=0)
            add_text_embeds = pooled_prompt_embeds
            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size,
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

            cutoff = num_inference_steps * adapter_conditioning_factor
            # 8. Denoising loop
            j = 0
            un = self.unet

            if do_self_attention_guidance:
                store_processor = sag.CrossAttnStoreProcessor()
                self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor  # type: ignore

            map_size = None

            def get_map_size(_, __, output):
                nonlocal map_size
                map_size = output[0].shape[
                    -2:
                ]  # output.sample.shape[-2:] in older diffusers

            classify = do_classifier_free_guidance

            def do_denoise(
                x: torch.Tensor,
                t: torch.IntTensor,
                call: Callable[..., torch.Tensor],
                change_source: Callable[[Callable], None],
            ) -> torch.Tensor:
                nonlocal j, un, do_classifier_free_guidance

                un = modify_kohya(un, j, num_inference_steps, deepshrink)  # type: ignore
                un = step_scalecrafter(un, scalecrafter, j, num_inference_steps)

                tau = j / num_inference_steps

                do_classifier_free_guidance = (
                    classify and tau <= config.api.cfg_uncond_tau
                )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([x] * 2) if do_classifier_free_guidance and not split_latents_into_two else x  # type: ignore
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

                if j >= refiner_steps:
                    assert refiner_model is not None
                    un = refiner_model

                down_intrablock_additional_residuals = None
                if hasattr(self, "adapter") and self.adapter is not None:
                    if j < cutoff:
                        assert adapter_state is not None
                        down_intrablock_additional_residuals = [
                            state.clone() for state in adapter_state
                        ]

                change_source(un)
                kwargs = set(
                    inspect.signature(un.forward).parameters.keys()  # type: ignore
                )
                ensure_correct_device(un)  # type: ignore

                _kwargs = {
                    "added_cond_kwargs": {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    },
                    "down_intrablock_additional_residuals": down_intrablock_additional_residuals,
                    "order": j,
                    "drop_encode_decode": config.api.drop_encode_decode != "off",
                }
                if split_latents_into_two:
                    uncond, cond = prompt_embeds.chunk(2)
                    uncond_text, cond_text = add_text_embeds.chunk(2)
                    uncond_time, cond_time = add_time_ids.chunk(2)
                    uncond_intra, cond_intra = None, None
                    if down_intrablock_additional_residuals is not None:
                        uncond_intra, cond_intra = [], []
                        for s in down_intrablock_additional_residuals:
                            unc, cnd = s.chunk(2)
                            uncond_intra.append(unc)
                            cond_intra.append(cnd)

                    added_cond_kwargs = {
                        "text_embeds": cond_text,
                        "time_ids": cond_time,
                    }
                    added_uncond_kwargs = {
                        "text_embeds": uncond_text,
                        "time_ids": uncond_time,
                    }

                    _kwargs.update(
                        {
                            "added_cond_kwargs": added_cond_kwargs,
                            "down_intrablock_additional_residuals": cond_intra,
                        }
                    )
                    for kw, _ in _kwargs.copy().items():
                        if kw not in kwargs:
                            del _kwargs[kw]
                    noise_pred_text = call(latent_model_input, t, cond=cond, **_kwargs)

                    _kwargs.update(
                        {
                            "added_cond_kwargs": added_uncond_kwargs,
                            "down_intrablock_additional_residuals": uncond_intra,
                        }
                    )
                    for kw, _ in _kwargs.copy().items():
                        if kw not in kwargs:
                            del _kwargs[kw]
                    noise_pred_uncond = call(
                        latent_model_input, t, cond=uncond, **_kwargs
                    )
                else:
                    for kw, _ in _kwargs.copy().items():
                        if kw not in kwargs:
                            del _kwargs[kw]
                    noise_pred = call(latent_model_input, t, cond=prompt_embeds, **_kwargs)  # type: ignore

                un, noise_pred_vanilla = post_scalecrafter(
                    self.unet,
                    scalecrafter,
                    j,
                    num_inference_steps,
                    call,
                    latent_model_input,
                    t,
                    cond=prompt_embeds,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    if not split_latents_into_two:
                        if isinstance(noise_pred, list):
                            noise_pred = noise_pred[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # type: ignore
                    noise_pred = calculate_cfg(
                        j, noise_pred_text, noise_pred_uncond, guidance_scale, t, additional_pred=noise_pred_vanilla  # type: ignore
                    )

                if do_self_attention_guidance:
                    if not do_classifier_free_guidance:
                        noise_pred_uncond = noise_pred  # type: ignore
                    noise_pred += sag.calculate_sag(  # type: ignore
                        self,
                        call,
                        store_processor,  # type: ignore
                        x,
                        noise_pred_uncond,  # type: ignore
                        t,
                        map_size,  # type: ignore
                        prompt_embeds,
                        self_attention_scale,
                        guidance_scale,
                        config.api.load_dtype,
                    )

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
                j += 1
                un = postprocess_kohya(un)
                return x

            if do_self_attention_guidance:
                pass

            ensure_correct_device(self.unet)
            latents = latents.to(dtype=dtype)  # type: ignore
            if init_latents_orig is not None:
                init_latents_orig = init_latents_orig.to(dtype=dtype)
            with ExitStack() as gs:
                if do_self_attention_guidance:
                    gs.enter_context(self.unet.mid_block.attentions[0].register_forward_hook(get_map_size))  # type: ignore

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
                        ret = s(
                            *args,
                            encoder_hidden_states=encoder_hidden_states,  # type: ignore
                            return_dict=False,
                            **kwargs,
                        )
                        if isinstance(s, UNet2DConditionModel):
                            return ret[0]
                        return ret

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
            if output_type == "latent":
                unload_all()
                return latents, False

            image = full_vae(latents, self.vae, height=height, width=width)  # type: ignore

            # 11. Convert to PIL
            if output_type == "pil":
                image = numpy_to_pil(image)  # type: ignore

            unload_all()

            if not return_dict:
                return image, False

            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=False  # type: ignore
            )
