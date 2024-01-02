# AnimateDiff is NCFHW (batch, channels, frames, height, width)

from typing import Optional

import numpy as np

from .freeinit import freq_mix_3d as freeinit_mix, get_freq_filter as freeinit_filter
from .models.unet import UNet3DConditionModel
from .pia.masking import prepare_mask_coef_by_statistics


def ordered_halving(val):
    "Returns fraction that has denominator that is a power of 2"

    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)
    final = as_int / (1 << 64)
    return final


# Generator that returns lists of latent indeces to diffuse on
def uniform(
    step: int = 0,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:  # type: ignore
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1  # type: ignore
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


def uniform_v2(
    step: int = 0,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:  # type: ignore
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1  # type: ignore
    )

    pad = int(round(num_frames * ordered_halving(step)))
    for context_step in 1 << np.arange(context_stride):
        j_initial = int(ordered_halving(step) * context_step) + pad
        for j in range(
            j_initial,
            num_frames + pad - context_overlap,
            (context_size * context_step - context_overlap),
        ):
            if context_size * context_step > num_frames:
                # On the final context_step,
                # ensure no frame appears in the window twice
                yield [e % num_frames for e in range(j, j + num_frames, context_step)]
                continue
            j = j % num_frames
            if j > (j + context_size * context_step) % num_frames and not closed_loop:
                yield [e for e in range(j, num_frames, context_step)]
                j_stop = (j + context_size * context_step) % num_frames
                # When  ((num_frames % (context_size - context_overlap)+context_overlap) % context_size != 0,
                # This can cause 'superflous' runs where all frames in
                # a context window have already been processed during
                # the first context window of this stride and step.
                # While the following commented if should prevent this,
                # I believe leaving it in is more correct as it maintains
                # the total conditional passes per frame over a large total steps
                # if j_stop > context_overlap:
                yield [e for e in range(0, j_stop, context_step)]
                continue
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


def uniform_constant(
    step: int = 0,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:  # type: ignore
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)  # type: ignore

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            skip_this_window = False
            prev_val = -1
            to_yield = []
            for e in range(j, j + context_size * context_step, context_step):
                e = e % num_frames
                if not closed_loop and e < prev_val:
                    skip_this_window = True
                    break
                to_yield.append(e)
                prev_val = e
            if skip_this_window:
                continue
            yield to_yield


def get_context_scheduler(name: str):
    return globals().get(name, nil_scheduler)


def nil_scheduler(*args, **kwargs):
    yield 0