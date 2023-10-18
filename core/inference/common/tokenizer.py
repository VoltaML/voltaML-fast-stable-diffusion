from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import logging

import torch
import PIL.Image

from ..utilities.philox import PhiloxGenerator

logger = logging.getLogger(__name__)


class CommonMultimodalInterpreter(ABC):
    """
    Abstract class for handling all things input; including, but not limited to:
    interpolation between multiple inputs, parsing of image, mask and text.
    """

    @abstractmethod
    def interpolate(
        self,
        images_and_prompts: List[Any],
        weights: List[float],
        generator: Union[torch.Generator, PhiloxGenerator],
    ):
        """Interpolate between images and prompts with the given weights."""

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
    ):
        """Encode prompt"""

    @abstractmethod
    def encode_image(self, image: PIL.Image.Image):
        """Encode image"""

    @abstractmethod
    def encode_mask(
        self,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
    ):
        """Encode mask"""
