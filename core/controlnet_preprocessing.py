import gc
import logging
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.upernet import UperNetForSemanticSegmentation

from core import shared_dependent
from core.config import config
from core.controlnet_utils import ade_palette
from core.types import ControlNetData

logger = logging.getLogger(__name__)

try:
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        MidasDetector,
        MLSDdetector,
        OpenposeDetector,
    )
except ImportError:
    logger.warning(
        "You have old version of controlnet-aux, please run `pip uninstall controlnet-aux && pip install controlnet-aux` to update it to the lates version."
    )


def wipe_old():
    "Wipes old controlnet preprocessor from memory"

    logger.debug("Did not find this controlnet preprocessor cached, wiping old ones")
    shared_dependent.cached_controlnet_preprocessor = None

    if config.api.device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def cache_preprocessor(preprocessor: Any):
    "Caches controlnet preprocessor"

    logger.debug(
        f"Caching {preprocessor.__class__.__name__ if not isinstance(preprocessor, tuple) else preprocessor[0].__class__.__name__ + ' + ' + preprocessor[1].__class__.__name__} preprocessor"
    )
    shared_dependent.cached_controlnet_preprocessor = preprocessor


def image_to_controlnet_input(
    input_image: Image.Image,
    data: ControlNetData,
) -> Image.Image:
    "Converts an image to the format expected by the controlnet model"

    model = data.controlnet

    if model == "none":
        return input_image
    elif "canny" in model:
        return canny(
            input_image,
            low_threshold=data.canny_low_threshold,
            high_threshold=data.canny_low_threshold * 3,
        )
    elif "depth" in model:
        return depth(input_image)
    elif "hed" in model:
        return hed(
            input_image,
            detect_resolution=data.detection_resolution,
            image_resolution=min(input_image.size),
        )
    elif "mlsd" in model:
        return mlsd(
            input_image,
            resolution=data.detection_resolution,
            score_thr=data.mlsd_thr_v,
            dist_thr=data.mlsd_thr_d,
        )
    elif "normal" in model:
        return normal(input_image)
    elif "openpose" in model:
        return openpose(input_image)
    elif "scribble" in model:
        return scribble(input_image)
    elif "seg" in model:
        return segmentation(input_image)

    raise NotImplementedError


def canny(
    input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    "Applies canny edge detection to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, CannyDetector):
        detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        detector = CannyDetector()
        cache_preprocessor(detector)

    canny_image = detector(
        img=input_image, low_threshold=low_threshold, high_threshold=high_threshold
    )

    return canny_image


def depth(input_image: Image.Image) -> Image.Image:
    "Applies depth estimation to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, MidasDetector):
        midas_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        midas_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        cache_preprocessor(midas_detector)

    image = midas_detector(input_image)

    if isinstance(image, tuple):
        return image[0]
    else:
        return image


def hed(
    input_image: Image.Image, detect_resolution=512, image_resolution=512
) -> Image.Image:
    "Applies hed edge detection to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, HEDdetector):
        hed_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        hed_detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        cache_preprocessor(hed_detector)

    image = hed_detector(
        input_image,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
    )

    assert isinstance(image, Image.Image)
    return image


def mlsd(
    input_image: Image.Image,
    resolution: int = 512,
    score_thr: float = 0.1,
    dist_thr: float = 20,
) -> Image.Image:
    "Applies M-LSD edge detection to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, MLSDdetector):
        mlsd_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        cache_preprocessor(mlsd_detector)

    image = mlsd_detector(
        input_image,
        thr_v=score_thr,
        thr_d=dist_thr,
        detect_resolution=resolution,
        image_resolution=resolution,
    )

    assert isinstance(image, Image.Image)
    return image


def normal(input_image: Image.Image) -> Image.Image:
    "Applies normal estimation to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, MidasDetector):
        midas_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        midas_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        cache_preprocessor(midas_detector)

    image = midas_detector(input_image, depth_and_normal=True)  # type: ignore

    if isinstance(image, tuple):
        return image[1]
    else:
        raise ValueError("MidasDetector did not return a tuple")


def openpose(input_image: Image.Image) -> Image.Image:
    "Applies openpose to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, OpenposeDetector):
        op_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        op_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        cache_preprocessor(op_detector)

    image = op_detector(input_image)

    assert isinstance(image, Image.Image)
    return image


def scribble(input_image: Image.Image) -> Image.Image:
    "Applies scribble to an image"

    return input_image


def segmentation(input_image: Image.Image) -> Image.Image:
    "Applies segmentation to an image"

    if isinstance(shared_dependent.cached_controlnet_preprocessor, Tuple):
        (  # pylint: disable=unpacking-non-sequence
            image_processor,
            image_segmentor,
        ) = shared_dependent.cached_controlnet_preprocessor
    else:
        wipe_old()
        image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        cache_preprocessor((image_processor, image_segmentor))

    pixel_values = image_processor(input_image, return_tensors="pt").pixel_values

    assert isinstance(image_segmentor, UperNetForSemanticSegmentation)
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[input_image.size[::-1]]
    )[0]

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3

    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)
    return image
