# pylint: disable=attribute-defined-outside-init

from dataclasses import dataclass
from typing import Literal, Union, List
from time import time
import math
import regex as re

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    SchedulerMixin,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from tqdm import tqdm
import numpy as np
from numpy import pi, exp, sqrt
from PIL.Image import Image

from core.config import config
from core.optimizations import autocast
from core.inference.utilities import preprocess_image, get_weighted_text_embeddings
from core.utils import resize

mask = Literal["constant", "gaussian", "quartic"]
reroll = Literal["reset", "epsilon"]


@dataclass
class CanvasRegion:
    """Class defining a region in the canvas."""

    row_init: int
    row_end: int
    col_init: int
    col_end: int
    region_seed: int = None  # type: ignore
    noise_eps: float = 0.0

    def __post_init__(self):
        if self.region_seed is None:
            self.region_seed = math.ceil(time())
        coords = [self.row_init, self.row_end, self.col_init, self.col_end]
        for coord in coords:
            if coord < 0:
                raise ValueError(
                    f"Region coordinates must be non-negative, found {coords}."
                )
            if coord % 8 != 0:
                raise ValueError(
                    f"Region coordinates must be multiples of 8, found {coords}."
                )
        if self.noise_eps < 0:
            raise ValueError(
                f"Region epsilon must be non-negative, found {self.noise_eps}."
            )
        self.latent_row_init = self.row_init // 8
        self.latent_row_end = self.row_end // 8
        self.latent_col_init = self.col_init // 8
        self.latent_col_end = self.col_end // 8

    @property
    def width(self):
        "col_end - col_init"
        return self.col_end - self.col_init

    @property
    def height(self):
        "row_end - row_init"
        return self.row_end - self.row_init

    def get_region_generator(self, device: Union[torch.device, str] = "cpu"):
        """Creates a generator for the region."""
        return torch.Generator(device).manual_seed(self.region_seed)


@dataclass
class DiffusionRegion(CanvasRegion):
    """Abstract class for places where diffusion is taking place."""


@dataclass
class RerollRegion(CanvasRegion):
    """Class defining a region where latent rerolling will be taking place."""

    reroll_mode: reroll = "reset"


@dataclass
class Text2ImageRegion(DiffusionRegion):
    """Class defining a region where text guided diffusion will be taking place."""

    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    mask_type: mask = "gaussian"
    mask_weight: float = 1.0

    text_embeddings = None

    def __post_init__(self):
        super().__post_init__()
        if self.mask_weight < 0:
            raise ValueError(
                f"Mask weight must be non-negative, found {self.mask_weight}."
            )
        self.prompt = re.sub(" +", " ", self.prompt).replace("\n", " ")

    @property
    def do_classifier_free_guidance(self) -> bool:
        """Whether to do classifier-free guidance (guidance_scale > 1.0)"""
        return self.guidance_scale > 1.0


@dataclass
class Image2ImageRegion(DiffusionRegion):
    """Class defining a region where image guided diffusion will be taking place."""

    reference_image: Image = None  # type: ignore
    strength: float = 0.8

    def __post_init__(self):
        super().__post_init__()
        if self.reference_image is None:
            raise ValueError("Reference image must be provided.")
        if self.strength < 0 or self.strength > 1:
            raise ValueError(
                f"Strength must be between 0 and 1, found {self.strength}."
            )
        self.reference_image = resize(self.reference_image, self.width, self.height)

    def encode_reference_image(
        self,
        encoder,
        generator: torch.Generator,
        device: Union[torch.device, str] = "cpu",
    ):
        """Encodes the reference image for this diffusion region."""
        img = preprocess_image(self.reference_image)
        self.reference_latents = encoder.encode(img.to(device)).latent_dist.sample(
            generator=generator
        )
        self.reference_latents = 0.18215 * self.reference_latents


@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region."""

    latent_space_dim: int
    nbatch: int = 1

    def compute_mask_weights(self, region: Text2ImageRegion) -> torch.Tensor:
        """Computes a tensor of weights for the given diffusion region."""
        if region.mask_type == "gaussian":
            mask_weights = self._gaussian_weights(region)
        elif region.mask_type == "constant":
            mask_weights = self._constant_weights(region)
        else:
            mask_weights = self._quartic_weights(region)
        return mask_weights

    def _constant_weights(self, region: Text2ImageRegion) -> torch.Tensor:
        """Computes a tensor of constant weights for the given diffusion region."""
        return (
            torch.ones(
                (
                    self.nbatch,
                    self.latent_space_dim,
                    region.latent_row_end - region.latent_row_init,
                    region.latent_col_end - region.latent_col_init,
                )
            )
            * region.mask_weight
        )

    def _gaussian_weights(self, region: Text2ImageRegion) -> torch.Tensor:
        """Computes a tensor of gaussian weights for the given diffusion region."""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01
        midpoint = (
            latent_width - 1
        ) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(
                -(x - midpoint)
                * (x - midpoint)
                / (latent_width * latent_width)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = (latent_height - 1) / 2
        y_probs = [
            exp(
                -(y - midpoint)
                * (y - midpoint)
                / (latent_height * latent_height)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(
            torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1)
        )

    def _quartic_weights(self, region: Text2ImageRegion) -> torch.Tensor:
        """Generates a quartic mask of weights for tile contributions

        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
        quartic_constant = 15.0 / 16.0

        support = (
            np.array(range(region.latent_col_init, region.latent_col_end))
            - region.latent_col_init
        ) / (region.latent_col_end - region.latent_col_init - 1) * 1.99 - (1.99 / 2.0)
        x_probs = quartic_constant * np.square(1 - np.square(support))
        support = (
            np.array(range(region.latent_row_init, region.latent_row_end))
            - region.latent_row_init
        ) / (region.latent_row_end - region.latent_row_init - 1) * 1.99 - (1.99 / 2.0)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(
            torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1)
        )


class StableDiffusionCanvasPipeline(DiffusionPipeline):
    """Stable Diffusion pipeline with support for multiple diffusions on one canvas."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

    def _decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return self.numpy_to_pil(image)

    def _get_timesteps(self, num_inference_steps: int, strength: float) -> torch.Tensor:
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * (1 - strength)) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = min(
            max(num_inference_steps - init_timestep + offset, 0),
            num_inference_steps - 1,
        )
        latest_timestep = self.scheduler.timesteps[t_start]
        return latest_timestep

    @property
    def _execution_device(self):
        # TODO: implement this from the SDXL PR
        return self.unet.device

    def __call__(
        self,
        height: int,
        width: int,
        regions: List[DiffusionRegion],
        generator: torch.Generator,
        num_inference_steps: int = 50,
        reroll_regions: List[RerollRegion] = [],
    ) -> List[Image]:
        batch_size = 1
        device = self._execution_device
        unet_channels = self.unet.config.in_channels

        txt_regions = [r for r in regions if isinstance(r, Text2ImageRegion)]
        img_regions = [r for r in regions if isinstance(r, Image2ImageRegion)]

        latents_shape = (
            batch_size,
            unet_channels,
            math.ceil(height / 8),
            math.ceil(width / 8),
        )

        all_eps_rerolls = regions + [
            r for r in reroll_regions if r.reroll_mode == "epsilon"
        ]

        with autocast(
            dtype=self.unet.dtype,
            disable=not config.api.autocast,
        ):
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            for region in txt_regions:
                prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(
                    self,  # type: ignore
                    region.prompt,
                    region.negative_prompt,
                    3,
                    False,
                    False,
                    False,
                )
                if region.do_classifier_free_guidance:
                    region.text_embeddings = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)  # type: ignore
                else:
                    region.text_embeddings = prompt_embeds

            init_noise = torch.randn(latents_shape, device=device, generator=generator)

            for region in reroll_regions:
                if region.reroll_mode == "reset":
                    latent_height = region.latent_row_end - region.latent_row_init
                    latent_width = region.latent_col_end - region.latent_col_init
                    region_shape = (
                        latents_shape[0],
                        latents_shape[1],
                        latent_height,
                        latent_width,
                    )
                    init_noise[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ] = torch.randn(region_shape, device=device, generator=generator)

            for region in all_eps_rerolls:
                if region.noise_eps > 0:
                    region_noise = init_noise[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ]
                    eps_noise = (
                        torch.randn(
                            region_noise.shape,
                            generator=region.get_region_generator(device=device),
                            device=device,
                        )
                        * region.noise_eps
                    )
                    init_noise[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ] += eps_noise

            latents = init_noise * self.scheduler.init_noise_sigma

            for region in img_regions:
                region.encode_reference_image(
                    self.vae, generator=generator, device=device
                )

            mask_builder = MaskWeightsBuilder(unet_channels, batch_size)
            mask_weights = [
                mask_builder.compute_mask_weights(r).to(device=device)
                for r in txt_regions
            ]
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                noise_pred_regions = []

                for region in txt_regions:
                    region_latents = latents[
                        :,
                        :,
                        region.latent_row_end : region.latent_row_init,
                        region.latent_col_init : region.latent_col_end,
                    ]
                    latent_model_input = region_latents
                    if region.do_classifier_free_guidance:
                        latent_model_input = torch.cat([latent_model_input] * 2)
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=region.text_embeddings,
                    ).sample
                    if region.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_region = (
                            noise_pred_uncond
                            + region.guidance_scale
                            * (noise_pred_text - noise_pred_uncond)
                        )
                        noise_pred_regions.append(noise_pred_region)
                noise_pred = torch.zeros(latents.shape, device=device)
                contributors = torch.zeros(latents.shape, device=device)

                for region, noise_pred_region, mask_weights_region in zip(
                    txt_regions, noise_pred_regions, mask_weights
                ):
                    noise_pred[
                        :,
                        :,
                        region.latent_row_end : region.latent_row_init,
                        region.latent_col_init : region.latent_col_end,
                    ] += (
                        noise_pred_region * mask_weights_region
                    )
                    contributors[
                        :,
                        :,
                        region.latent_row_end : region.latent_row_init,
                        region.latent_col_init : region.latent_col_end,
                    ] += mask_weights_region
                noise_pred /= contributors
                noise_pred = torch.nan_to_num(noise_pred)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                for region in img_regions:
                    influence_step = self._get_timesteps(
                        num_inference_steps, region.strength
                    )
                    if t > influence_step:
                        timestep = t.repeat(batch_size)
                        region_init_noise = init_noise[
                            :,
                            :,
                            region.latent_row_init : region.latent_row_end,
                            region.latent_col_init : region.latent_col_end,
                        ]
                        region_latents = self.scheduler.add_noise(
                            region.reference_latents, region_init_noise, timestep
                        )
                        latents[
                            :,
                            :,
                            region.latent_row_init : region.latent_row_end,
                            region.latent_col_init : region.latent_col_end,
                        ] = region_latents
            image = self._decode_latents(latents)
            return image
