import gc
import logging
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.upernet import UperNetForSemanticSegmentation

from core.config import config
from core.types import ControlNetData

logger = logging.getLogger(__name__)

try:
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        LineartAnimeDetector,
        LineartDetector,
        MidasDetector,
        MLSDdetector,
        NormalBaeDetector,
        OpenposeDetector,
    )
except ImportError:
    logger.warning(
        "You have old version of controlnet-aux, please run `pip uninstall controlnet-aux && pip install controlnet-aux` to update it to the lates version."
    )


def _wipe_old():
    "Wipes old controlnet preprocessor from memory"

    from core import shared_dependent

    logger.debug("Did not find this controlnet preprocessor cached, wiping old ones")
    shared_dependent.cached_controlnet_preprocessor = None

    if config.api.device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _cache_preprocessor(preprocessor: Any):
    "Caches controlnet preprocessor"

    from core import shared_dependent

    logger.debug(
        f"Caching {preprocessor.__class__.__name__ if not isinstance(preprocessor, tuple) else preprocessor[0].__class__.__name__ + ' + ' + preprocessor[1].__class__.__name__} preprocessor"
    )
    shared_dependent.cached_controlnet_preprocessor = preprocessor


def image_to_controlnet_input(
    input_image: Image.Image,
    data: ControlNetData,
) -> Image.Image:
    "Converts an image to the format expected by the controlnet model"

    model = data.controlnet.split("/")[-1]

    if any(item in model for item in ["none", "scribble", "ip2p", "tile", "qrcode"]):
        return input_image
    elif "canny" in model:
        return _canny(
            input_image,
            low_threshold=data.canny_low_threshold,
            high_threshold=data.canny_low_threshold * 3,
        )
    elif "depth" in model:
        return _midas(input_image)
    elif any(item in model for item in ["hed", "softedge"]):
        return _hed(
            input_image,
            detect_resolution=data.detection_resolution,
            image_resolution=min(input_image.size),
        )
    elif "mlsd" in model:
        return _mlsd(
            input_image,
            resolution=data.detection_resolution,
            score_thr=data.mlsd_thr_v,
            dist_thr=data.mlsd_thr_d,
        )
    elif "normal" in model:
        return _normal_bae(input_image)
    elif "openpose" in model:
        return _openpose(input_image)
    elif "seg" in model:
        return _segmentation(input_image)

    raise NotImplementedError


def _canny(
    input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    "Applies canny edge detection to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, CannyDetector):
        detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        detector = CannyDetector()
        _cache_preprocessor(detector)

    canny_image = detector(
        img=input_image, low_threshold=low_threshold, high_threshold=high_threshold
    )

    return canny_image


def _midas(input_image: Image.Image) -> Image.Image:
    "Applies depth estimation to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, MidasDetector):
        midas_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        midas_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(midas_detector)

    image = midas_detector(input_image)

    if isinstance(image, tuple):
        return image[0]
    else:
        return image


def _hed(
    input_image: Image.Image, detect_resolution=512, image_resolution=512
) -> Image.Image:
    "Applies hed edge detection to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, HEDdetector):
        hed_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(hed_detector)

    image = hed_detector(
        input_image,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
    )

    assert isinstance(image, Image.Image)
    return image


def _mlsd(
    input_image: Image.Image,
    resolution: int = 512,
    score_thr: float = 0.1,
    dist_thr: float = 20,
) -> Image.Image:
    "Applies M-LSD edge detection to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, MLSDdetector):
        mlsd_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(mlsd_detector)

    image = mlsd_detector(
        input_image,
        thr_v=score_thr,
        thr_d=dist_thr,
        detect_resolution=resolution,
        image_resolution=resolution,
    )

    assert isinstance(image, Image.Image)
    return image


def _normal_bae(input_image: Image.Image) -> Image.Image:
    "Applies normal estimation to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, NormalBaeDetector):
        normal_bae_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        normal_bae_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(normal_bae_detector)

    image = normal_bae_detector(input_image, depth_and_normal=True)  # type: ignore

    return image


def _openpose(input_image: Image.Image) -> Image.Image:
    "Applies openpose to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, OpenposeDetector):
        op_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        op_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(op_detector)

    image = op_detector(input_image, hand_and_face=True)

    assert isinstance(image, Image.Image)
    return image


def _segmentation(input_image: Image.Image) -> Image.Image:
    "Applies segmentation to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, Tuple):
        (  # pylint: disable=unpacking-non-sequence
            image_processor,
            image_segmentor,
        ) = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        _cache_preprocessor((image_processor, image_segmentor))

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

    palette = np.array(_ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)
    return image


def _lineart(input_image: Image.Image) -> Image.Image:
    "Applies lineart to an image"

    from core import shared_dependent

    if isinstance(shared_dependent.cached_controlnet_preprocessor, LineartDetector):
        op_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        op_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(op_detector)

    image = op_detector(input_image)

    assert isinstance(image, Image.Image)
    return image


def _lineart_anime(input_image: Image.Image) -> Image.Image:
    "Applies lineart_anime to an image"

    from core import shared_dependent

    if isinstance(
        shared_dependent.cached_controlnet_preprocessor, LineartAnimeDetector
    ):
        op_detector = shared_dependent.cached_controlnet_preprocessor
    else:
        _wipe_old()
        op_detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        _cache_preprocessor(op_detector)

    image = op_detector(input_image)

    assert isinstance(image, Image.Image)
    return image


def _ade_palette():
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
