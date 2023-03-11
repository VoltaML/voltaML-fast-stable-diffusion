import cv2
import numpy as np
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from PIL import Image

from core.types import ControlNetData, ControlNetMode


def image_to_controlnet_input(
    input_image: Image.Image,
    data: ControlNetData,
) -> Image.Image:
    "Converts an image to the format expected by the controlnet model"

    model = data.controlnet

    if model == ControlNetMode.NONE:
        return input_image
    elif model == ControlNetMode.CANNY:
        return canny(
            input_image,
            low_threshold=data.canny_low_threshold,
            high_threshold=data.canny_low_threshold * 3,
        )
    elif model == ControlNetMode.DEPTH:
        return depth(input_image)
    elif model == ControlNetMode.HED:
        return hed(
            input_image,
            detect_resolution=data.detection_resolution,
            image_resolution=min(input_image.size),
        )
    elif model == ControlNetMode.MLSD:
        return mlsd(
            input_image,
            resolution=data.detection_resolution,
            thr_v=data.mlsd_thr_v,
            thr_d=data.mlsd_thr_d,
        )
    elif model == ControlNetMode.NORMAL:
        return normal(input_image)
    elif model == ControlNetMode.OPENPOSE:
        return openpose(input_image)
    elif model == ControlNetMode.SCRIBBLE:
        return scribble(input_image)
    elif model == ControlNetMode.SEGMENTATION:
        return segmentation(input_image)

    raise NotImplementedError


def canny(
    input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    "Applies canny edge detection to an image"

    image = np.array(input_image)

    image = cv2.Canny(image, low_threshold, high_threshold)  # pylint: disable=no-member
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image


def depth(input_image: Image.Image) -> Image.Image:
    "Applies depth estimation to an image"

    raise NotImplementedError


def hed(
    input_image: Image.Image, detect_resolution=512, image_resolution=512
) -> Image.Image:
    "Applies hed edge detection to an image"

    hed_detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")
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
    thr_v: float = 0.1,
    thr_d: float = 0.1,
) -> Image.Image:
    "Applies M-LSD edge detection to an image"

    mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    image = mlsd_detector(
        input_image,
        thr_v=thr_v,
        thr_d=thr_d,
        detect_resolution=resolution,
        image_resolution=resolution,
    )

    assert isinstance(image, Image.Image)
    return image


def normal(input_image: Image.Image) -> Image.Image:
    "Applies normal estimation to an image"

    raise NotImplementedError


def openpose(input_image: Image.Image) -> Image.Image:
    "Applies openpose to an image"

    op_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = op_detector(input_image)

    assert isinstance(image, Image.Image)
    return image


def scribble(input_image: Image.Image) -> Image.Image:
    "Applies scribble to an image"

    raise NotImplementedError


def segmentation(input_image: Image.Image) -> Image.Image:
    "Applies segmentation to an image"

    raise NotImplementedError
