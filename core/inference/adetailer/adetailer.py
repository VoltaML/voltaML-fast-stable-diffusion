# Taken from https://github.com/Bing-su/asdff
# Origial author: Bing-su
# Modified by: Stax124

import functools
import logging
from typing import Any, Callable, Iterable, List, Optional

from asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from asdff.yolo import yolo_detector
from PIL import Image, ImageOps

from core.types import InpaintQueueEntry
from core.utils import convert_to_image

logger = logging.getLogger(__name__)

DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class ADetailer:
    def get_default_detector(self, model_path: Optional[str] = None):
        if model_path is not None:
            return functools.partial(yolo_detector, model_path=model_path)

        return yolo_detector

    def generate(
        self,
        fn: Any,
        inpaint_entry: InpaintQueueEntry,
        detectors: DetectorType | Iterable[DetectorType] | None = None,
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32,
        iterations: int = 1,
        upscale: int = 2,
        yolo_model: Optional[str] = None,
    ) -> ADOutput:
        if detectors is None:
            detectors = [self.get_default_detector(yolo_model)]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        input_image = convert_to_image(inpaint_entry.data.image).convert("RGB")

        init_images = []
        final_images = []

        init_images.append(input_image.copy())
        final_image = None

        for j, detector in enumerate(detectors):
            masks = detector(input_image)
            if masks is None:
                logger.info(f"No object detected with {ordinal(j + 1)} detector.")
                continue

            for k, mask in enumerate(masks):
                mask = mask.convert("L")
                mask = mask_dilate(mask, mask_dilation)
                bbox = mask.getbbox()
                if bbox is None:
                    logger.info(f"No object in {ordinal(k + 1)} mask.")
                    continue
                mask = mask_gaussian_blur(mask, mask_blur)
                bbox_padded = bbox_padding(bbox, input_image.size, mask_padding)
                inverted_mask = ImageOps.invert(mask)

                for _i in range(iterations):
                    inpaint_output: List[Image.Image] = self.process_inpainting(
                        fn,
                        inpaint_entry,
                        input_image,
                        inverted_mask,
                        bbox_padded,
                        upscale=upscale,
                    )

                    inpaint_image: Image.Image = inpaint_output[0]  # type: ignore

                    final_image = composite(
                        input_image,
                        mask,
                        inpaint_image,
                        bbox_padded,
                    )

                    input_image = final_image

        assert final_image is not None
        final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    def process_inpainting(
        self,
        fn: Callable,
        inpaint_entry: InpaintQueueEntry,
        init_image: Image.Image,
        mask: Image.Image,
        bbox_padded: tuple[int, int, int, int],
        upscale: int = 2,
    ):  # -> tuple[PipelineImageInput, Any | None] | StableDiffusionPipelineOutput:
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)

        # Get the current size of the images
        width, height = crop_image.size

        # Calculate the new size
        new_size = (int(width * upscale), int(height * upscale))

        # Resize the images
        crop_image = crop_image.resize(new_size, resample=Image.LANCZOS)
        crop_mask = crop_mask.resize(new_size, resample=Image.LANCZOS)

        inpaint_entry.data.image = crop_image  # type: ignore
        inpaint_entry.data.mask_image = crop_mask  # type: ignore

        inpaint_entry.data.width = crop_image.width
        inpaint_entry.data.height = crop_image.height

        out: List[Image.Image] = fn(inpaint_entry)
        # Resize back to original size
        out_resized = []
        for image in out:
            out_resized.append(image.resize((width, height), resample=Image.LANCZOS))

        return out_resized
