# pylint: disable=attribute-defined-outside-init

from dataclasses import dataclass
from typing import Literal, Union
from time import time
import math
import regex as re

import torch
import numpy as np
from numpy import pi, exp, sqrt
from PIL.Image import Image

from core.inference.utilities import preprocess_image
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

    tokenized_prompt = None
    encoded_prompt = None

    def __post_init__(self):
        super().__post_init__()
        if self.mask_weight < 0:
            raise ValueError(
                f"Mask weight must be non-negative, found {self.mask_weight}."
            )
        self.prompt = re.sub(" +", " ", self.prompt).replace("\n", " ")

    def tokenize_prompt(self, tokenizer):
        """Tokenizes a prompt for this diffusion region."""
        self.tokenized_prompt = tokenizer(self.prompt, return_tensors="pt")


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
