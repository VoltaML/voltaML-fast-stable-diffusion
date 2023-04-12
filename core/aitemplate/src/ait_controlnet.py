import inspect
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from aitemplate.compiler import Model
from diffusers import AutoencoderKL, ControlNetModel, LMSDiscreteScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION, deprecate
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from core.aitemplate.config import get_unet_in_channels

logger = logging.getLogger(__name__)


class StableDiffusionControlNetAITPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

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
        controlnet ([`ControlNetModel`]):
            Provides additional conditioning to the unet during the denoising process
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        controlnet: ControlNetModel,
        directory: str = "",
        clip_ait_exe: Optional[Model] = None,
        unet_ait_exe: Optional[Model] = None,
        vae_ait_exe: Optional[Model] = None,
        requires_safety_checker: bool = True,
    ):
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "  # type: ignore
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)  # type: ignore
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)  # type: ignore

        if (
            hasattr(scheduler.config, "clip_sample")  # type: ignore
            and scheduler.config.clip_sample is True  # type: ignore
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)  # type: ignore
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)  # type: ignore

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.safety_checker: StableDiffusionSafetyChecker
        self.requires_safety_checker: bool
        self.feature_extractor: CLIPFeatureExtractor
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.safety_checker: StableDiffusionSafetyChecker
        self.feature_extractor: CLIPFeatureExtractor

        self.scheduler: LMSDiscreteScheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # type: ignore
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        logger.debug(f"AIT workdir: {directory}")

        if clip_ait_exe is None:
            self.clip_ait_exe = self.init_ait_module(
                model_name="CLIPTextModel", workdir=directory
            )
        else:
            self.clip_ait_exe = clip_ait_exe

        if unet_ait_exe is None:
            self.unet_ait_exe = self.init_ait_module(
                model_name="UNet2DConditionModel", workdir=directory
            )
        else:
            self.unet_ait_exe = unet_ait_exe

        if vae_ait_exe is None:
            self.vae_ait_exe = self.init_ait_module(
                model_name="AutoencoderKL", workdir=directory
            )
        else:
            self.vae_ait_exe = vae_ait_exe

        self.safety_checker: StableDiffusionSafetyChecker
        self.requires_safety_checker: bool
        self.feature_extractor: CLIPFeatureExtractor

        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.safety_checker: StableDiffusionSafetyChecker
        self.feature_extractor: CLIPFeatureExtractor
        self.controlnet: ControlNetModel

        self.scheduler: LMSDiscreteScheduler

        self.unet_in_channels = get_unet_in_channels(directory=Path(directory))

    def init_ait_module(
        self,
        model_name,
        workdir,
    ):
        mod = Model(os.path.join(workdir, model_name, "test.so"))
        return mod

    def unet_inference(
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        dbar_0,
        dbar_1,
        dbar_2,
        dbar_3,
        dbar_4,
        dbar_5,
        dbar_6,
        dbar_7,
        dbar_8,
        dbar_9,
        dbar_10,
        dbar_11,
        mid_block_additional_residual,
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
            "input3": dbar_0.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input4": dbar_1.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input5": dbar_2.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input6": dbar_3.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input7": dbar_4.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input8": dbar_5.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input9": dbar_6.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input10": dbar_7.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input11": dbar_8.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input12": dbar_9.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input13": dbar_10.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input14": dbar_11.permute((0, 2, 3, 1)).contiguous().cuda().half(),
            "input15": mid_block_additional_residual.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
        }
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2))
        return noise_pred

    def clip_inference(self, input_ids, seqlen=64):
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
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_outputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = exe_module.get_output_maximum_shape(i)
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

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[
            torch.FloatTensor,
            Image.Image,
            List[torch.FloatTensor],
            List[Image.Image],
        ],
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        controlnet_conditioning_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor`, `Image.Image`, `List[torch.FloatTensor]` or `List[Image.Image]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. Image.Image` can
                also be accepted as an image. The control image is automatically resized to fit the output image.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
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
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=64,  # self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input.input_ids.shape[-1]
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
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        init_image = self.prepare_image(
            image,
            width,
            height,
            batch_size * num_images_per_prompt,
            num_images_per_prompt,
            self.device,
            self.controlnet.dtype,
        )
        if do_classifier_free_guidance:
            init_image = torch.cat([init_image] * 2)

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet_in_channels, height // 8, width // 8)
        # import ipdb; ipdb.set_trace()
        if latents is None:
            latents = torch.randn(
                latents_shape,  # type: ignore
                generator=generator,
                device=latents_device,
            ).to(device=latents_device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        text_embeddings = text_embeddings.to(self.controlnet.dtype)
        latents = latents.to(device=self.device, dtype=text_embeddings.dtype)  # type: ignore
        latents = latents * self.scheduler.init_noise_sigma  # type: ignore

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # type: ignore
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=init_image,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet_inference(
                latent_model_input,
                t,
                text_embeddings,
                down_block_res_samples[0],
                down_block_res_samples[1],
                down_block_res_samples[2],
                down_block_res_samples[3],
                down_block_res_samples[4],
                down_block_res_samples[5],
                down_block_res_samples[6],
                down_block_res_samples[7],
                down_block_res_samples[8],
                down_block_res_samples[9],
                down_block_res_samples[10],
                down_block_res_samples[11],
                mid_block_res_sample,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs  # type: ignore
            ).prev_sample  # type: ignore

            if callback is not None:
                callback(i, t, latents)  # type: ignore

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents  # type: ignore
        init_image = self.vae_inference(latents)

        init_image = (init_image / 2 + 0.5).clamp(0, 1)
        init_image = init_image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(init_image), return_tensors="pt"
            ).to(self.device)
            init_image, has_nsfw_concept = self.safety_checker(
                images=init_image, clip_input=safety_checker_input.pixel_values
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            init_image = self.numpy_to_pil(init_image)

        if not return_dict:
            return (init_image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=init_image, nsfw_content_detected=has_nsfw_concept
        )
