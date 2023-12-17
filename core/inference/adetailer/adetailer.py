# Taken from https://github.com/Bing-su/asdff
# Origial author: Bing-su
# Modified by: Stax124

import functools
import inspect
import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional

from asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from asdff.yolo import yolo_detector
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from PIL import Image

logger = logging.getLogger(__name__)

DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class ADetailer:
    def __init__(
        self,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        unet: Any,
        scheduler: Any,
        safety_checker: Any,
        feature_extractor: Any = None,
        controlnet: Any = None,
        use_controlnet: bool = False,
    ):
        if not use_controlnet:
            self.inpaint_pipeline = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                requires_safety_checker=False,
            )
        else:
            self.inpaint_pipeline = StableDiffusionControlNetInpaintPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                requires_safety_checker=False,
            )

    def get_default_detector(self, model_path: Optional[str] = None):
        if model_path is not None:
            return functools.partial(yolo_detector, model_path=model_path)

        return yolo_detector

    def generate(
        self,
        input_images: List[Image.Image],
        detectors: DetectorType | Iterable[DetectorType] | None = None,
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32,
        common: Mapping[str, Any] | None = None,
        txt2img_only: Mapping[str, Any] | None = None,
        inpaint_only: Mapping[str, Any] | None = None,
        yolo_model: Optional[str] = None,
    ):
        if common is None:
            common = {}
        if txt2img_only is None:
            txt2img_only = {}
        if inpaint_only is None:
            inpaint_only = {}
        if "strength" not in inpaint_only:
            inpaint_only = {**inpaint_only, "strength": 0.4}

        if detectors is None:
            detectors = [self.get_default_detector(yolo_model)]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        init_images = []
        final_images = []

        for i, init_image in enumerate(input_images):
            init_images.append(init_image.copy())
            final_image = None

            for j, detector in enumerate(detectors):
                masks = detector(init_image)
                if masks is None:
                    logger.info(
                        f"No object detected on {ordinal(i + 1)} image with {ordinal(j + 1)} detector."
                    )
                    continue

                for k, mask in enumerate(masks):
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    bbox = mask.getbbox()
                    if bbox is None:
                        logger.info(f"No object in {ordinal(k + 1)} mask.")
                        continue
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)

                    inpaint_output = self.process_inpainting(
                        common,
                        inpaint_only,
                        init_image,
                        mask,
                        bbox_padded,
                    )
                    inpaint_image: Image.Image = inpaint_output[0][0]  # type: ignore

                    final_image = composite(
                        init_image,
                        mask,
                        inpaint_image,
                        bbox_padded,
                    )
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(init_images, final_images)

    def _get_inpaint_args(
        self, common: Mapping[str, Any], inpaint_only: Mapping[str, Any]
    ):
        common = dict(common)
        sig = inspect.signature(self.inpaint_pipeline)
        if (
            "control_image" in sig.parameters
            and "control_image" not in common
            and "image" in common
        ):
            common["control_image"] = common.pop("image")
        return {
            **common,
            **inpaint_only,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

    def process_inpainting(
        self,
        common: Mapping[str, Any],
        inpaint_only: Mapping[str, Any],
        init_image: Image.Image,
        mask: Image.Image,
        bbox_padded: tuple[int, int, int, int],
    ):  # -> tuple[PipelineImageInput, Any | None] | StableDiffusionPipelineOutput:
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
        inpaint_args = self._get_inpaint_args(common, inpaint_only)
        inpaint_args["image"] = crop_image
        inpaint_args["mask_image"] = crop_mask

        if "control_image" in inpaint_args:
            inpaint_args["control_image"] = inpaint_args["control_image"].resize(
                crop_image.size
            )
        return self.inpaint_pipeline(**inpaint_args)
