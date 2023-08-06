#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from typing import Optional

import numpy as np
import torch


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: torch.Tensor = image.flip(-3)
    # out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def auto_split_upscale(
    lr_img: np.ndarray,
    upscale_function,
    scale: int = 4,
    overlap: int = 32,
    max_depth: Optional[int] = None,
    current_depth: int = 1,
):
    # Attempt to upscale if unknown depth or if reached known max depth
    if max_depth is None or max_depth == current_depth:
        try:
            result = upscale_function(lr_img)
            return result, current_depth
        except RuntimeError as e:
            # Check to see if its actually the CUDA out of memory error
            if "CUDA" in str(e):
                # Collect garbage (clear VRAM)
                torch.cuda.empty_cache()
                gc.collect()
            # Re-raise the exception if not an OOM error
            else:
                raise RuntimeError(e) from e

    h, w, c = lr_img.shape

    # Split image into 4ths
    top_left = lr_img[: h // 2 + overlap, : w // 2 + overlap, :]
    top_right = lr_img[: h // 2 + overlap, w // 2 - overlap :, :]
    bottom_left = lr_img[h // 2 - overlap :, : w // 2 + overlap, :]
    bottom_right = lr_img[h // 2 - overlap :, w // 2 - overlap :, :]

    # Recursively upscale the quadrants
    # After we go through the top left quadrant, we know the maximum depth and no longer need to test for out-of-memory
    top_left_rlt, depth = auto_split_upscale(
        top_left,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=max_depth,
        current_depth=current_depth + 1,
    )
    top_right_rlt, _ = auto_split_upscale(
        top_right,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )
    bottom_left_rlt, _ = auto_split_upscale(
        bottom_left,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )
    bottom_right_rlt, _ = auto_split_upscale(
        bottom_right,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )

    # Define output shape
    out_h = h * scale
    out_w = w * scale

    # Create blank output image
    output_img = np.zeros((out_h, out_w, c), np.uint8)

    # Fill output image with tiles, cropping out the overlaps
    output_img[: out_h // 2, : out_w // 2, :] = top_left_rlt[
        : out_h // 2, : out_w // 2, :
    ]
    output_img[: out_h // 2, -out_w // 2 :, :] = top_right_rlt[
        : out_h // 2, -out_w // 2 :, :
    ]
    output_img[-out_h // 2 :, : out_w // 2, :] = bottom_left_rlt[
        -out_h // 2 :, : out_w // 2, :
    ]
    output_img[-out_h // 2 :, -out_w // 2 :, :] = bottom_right_rlt[
        -out_h // 2 :, -out_w // 2 :, :
    ]

    return output_img, depth
