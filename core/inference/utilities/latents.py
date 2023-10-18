import logging
import math
from time import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models import vae as diffusers_vae
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image

from core.config import config
from core.flags import LatentScaleModel
from core.inference.utilities.philox import PhiloxGenerator

from .random import randn

logger = logging.getLogger(__name__)


def _randn_tensor(
    shape: Union[Tuple, List],
    generator: Union[PhiloxGenerator, torch.Generator],
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,  # pylint: disable=unused-argument
):
    return randn(shape, generator, device, dtype)


diffusers_vae.randn_tensor = _randn_tensor
logger.debug("Overwritten diffusers randn_tensor")


def pad_tensor(
    tensor: torch.Tensor, multiple: int, size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    "Pad a tensors (NCHW) H and W dimension to the ceil(x / multiple)"
    batch_size, channels, height, width = tensor.shape
    new_height = math.ceil(height / multiple) * multiple
    new_width = math.ceil(width / multiple) * multiple
    hw = size or (new_height, new_width)
    if size or (new_width != width or new_height != height):
        nt = torch.zeros(
            batch_size,
            channels,
            hw[0],
            hw[1],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        nt[:, :, :height, :width] = tensor
        return nt
    else:
        return tensor


def prepare_mask_and_masked_image(
    image, mask, height: int, width: int, return_image: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The function resizes and converts input image and mask to PyTorch tensors,
    applies thresholding to mask tensor, obtains masked image tensor by multiplying image and mask tensors,
    and returns the resulting mask tensor, masked image tensor, and optionally the original image tensor.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            mask = [mask]
            mask = [i.resize((width, height), resample=Image.LANCZOS) for i in mask]
            mask = np.concatenate(
                [np.array(i.convert("L"))[None, None, :] for i in mask], axis=0
            )
            mask = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
            mask = pad_tensor(mask, 8)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            else:
                mask = mask.unsqueeze(1)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
        image = pad_tensor(image, 8)
    else:
        image = [image]
        mask = [mask]

        image = [i.resize((width, height), resample=Image.LANCZOS) for i in image]
        mask = [i.resize((width, height), resample=Image.LANCZOS) for i in mask]

        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        mask = np.concatenate(
            [np.array(i.convert("L"))[None, None, :] for i in mask], axis=0
        )

        image = np.concatenate(image, axis=0)
        mask = mask.astype(np.float32) / 255.0

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        image = pad_tensor(image, 8)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        mask = pad_tensor(mask, 8)

    masked_image = image * (mask < 0.5)
    if return_image:
        return mask, masked_image, image
    return mask, masked_image, None


def prepare_mask_latents(
    mask,
    masked_image,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    do_classifier_free_guidance: bool,
    vae,
    vae_scale_factor: float,
    vae_scaling_factor: float,
    generator: Union[PhiloxGenerator, torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function resizes and converts the input mask to a PyTorch tensor,
    encodes the input masked image to its latent representation,
    repeats the mask and masked image latents to match the batch size,
    concatenates the mask tensor if classifier-free guidance is enabled,
    and returns the resulting mask tensor and masked image latents."""
    mask = torch.nn.functional.interpolate(
        mask, size=(height // vae_scale_factor, width // vae_scale_factor)
    )
    mask = mask.to(device=device, dtype=dtype)

    masked_image = masked_image.to(device=device, dtype=dtype)
    masked_image_latents = vae_scaling_factor * vae.encode(
        masked_image
    ).latent_dist.sample(generator=generator)
    if mask.shape[0] < batch_size:
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        masked_image_latents = masked_image_latents.repeat(
            batch_size // masked_image_latents.shape[0], 1, 1, 1
        )
    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        torch.cat([masked_image_latents] * 2)
        if do_classifier_free_guidance
        else masked_image_latents
    )
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents


def preprocess_image(image):
    # w, h = image.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_image(
    image, width, height, batch_size, num_images_per_prompt, device, dtype
):
    "Prepare an image for controlnet 'consumption.'"
    if not isinstance(image, torch.Tensor):
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image[0], Image.Image):
            image = [
                np.array(
                    i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                )[None, :]
                for i in image
            ]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)  # type: ignore

    image_batch_size = image.shape[0]  # type: ignore

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)  # type: ignore

    image = image.to(device=device, dtype=dtype)

    return image


def preprocess_mask(mask):  # pylint: disable=unused-argument
    mask = mask.convert("L")
    # w, h = mask.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # mask = mask.resize(
    #     (w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"]
    # )
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    # Gabe: same as mask.unsqueeze(0)
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


def prepare_latents(
    pipe: StableDiffusionPipeline,
    image: Optional[torch.Tensor],
    timestep: torch.FloatTensor,
    batch_size: int,
    height: Optional[int],
    width: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
    generator: Union[PhiloxGenerator, torch.Generator],
    latents=None,
    latent_channels: Optional[int] = None,
    align_to: int = 1,
):
    if image is None:
        shape = (
            batch_size,
            pipe.unet.config.in_channels,  # type: ignore
            (math.ceil(height / align_to) * align_to) // pipe.vae_scale_factor,  # type: ignore
            (math.ceil(width / align_to) * align_to) // pipe.vae_scale_factor,  # type: ignore
        )

        if latents is None:
            # randn does not work reproducibly on mps
            latents = randn(shape, generator, device=device, dtype=dtype)  # type: ignore
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * pipe.scheduler.init_noise_sigma  # type: ignore
        return latents, None, None
    else:
        if pipe.vae.config.force_upcast or config.api.upcast_vae:
            from core.optimizations import upcast_vae

            upcast_vae(pipe.vae)

        if image.shape[1] != 4:
            image = pad_tensor(image, pipe.vae_scale_factor)
            init_latent_dist = pipe.vae.encode(image.to(config.api.device, dtype=pipe.vae.dtype)).latent_dist  # type: ignore
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents
            init_latents = torch.cat([init_latents] * batch_size, dim=0)
        else:
            logger.debug("Skipping VAE encode, already have latents")
            init_latents = image

        # if isinstance(pipe.scheduler, KdiffusionSchedulerAdapter):
        #     init_latents = init_latents * pipe.scheduler.init_noise_sigma

        init_latents_orig = init_latents
        shape = init_latents.shape
        if latent_channels is not None:
            shape = (
                batch_size,
                latent_channels,  # type: ignore
                (math.ceil(height / align_to) * align_to) // pipe.vae_scale_factor,  # type: ignore
                (math.ceil(width / align_to) * align_to) // pipe.vae_scale_factor,  # type: ignore
            )

        # add noise to latents using the timesteps
<<<<<<< HEAD
        if device.type == "mps" or config.api.device_type == "directml":
            noise = torch.randn(
                shape, generator=generator, device="cpu", dtype=dtype
            ).to(device)
        else:
            noise = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            ).to(device)
        latents = pipe.scheduler.add_noise(init_latents.to(device), noise, timestep.to(device))  # type: ignore
=======
        noise = randn(shape, generator, device=device, dtype=dtype)
        latents = pipe.scheduler.add_noise(init_latents.to(device), noise.to(device), timestep.to(device))  # type: ignore
>>>>>>> origin/experimental
        return latents, init_latents_orig, noise


def bislerp(samples, width, height):
    device = samples.device

    def slerp(b1, b2, r):
        c = b1.shape[-1]
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # Normalize b1 and b2
        b1_normalized = b1.div_(b1_norms)
        b2_normalized = b2.div_(b2_norms)

        # Handle zero norms
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # Calculate the dot product of b1_normalized and b2_normalized
        dot = (b1_normalized * b2_normalized).sum(1)

        # Clamp dot to the valid range of [-1, 1]
        dot = torch.clamp(dot, -1.0, 1.0)

        # Calculate the angle between b1_normalized and b2_normalized
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # Calculate the spherical interpolation of b1_normalized and b2_normalized
        epsilon = 1e-3  # maybe change to smaller value (not too small though, nans happen here and there on 1e-8)
        soep = so + epsilon
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / soep).unsqueeze(
            1
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / soep).unsqueeze(
            1
        ) * b2_normalized

        if torch.isnan(res).any():
            res = torch.nan_to_num(res)

        # Multiply the result by the linear interpolation of b1_norms and b2_norms
        res.mul_((b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c))

        # If dot is very close to 1 (almost parallel vectors), use b1 directly as the result
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5].to(res.dtype)

        # If dot is very close to -1 (almost antiparallel vectors), use the linear interpolation of b1 and b2 as the result
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1].to(res.dtype)
        return res

    def generate_bilinear_data(length_old, length_new):
        # Create a tensor of evenly spaced coordinates along the old axis
        coords_1 = torch.arange(length_old, device=device, dtype=torch.float32).view(
            1, 1, 1, -1
        )

        # Interpolate the coordinates to the new axis
        coords_1 = F.interpolate(
            coords_1, size=(1, length_new), mode="bilinear", align_corners=False
        )

        # Calculate the interpolation ratios
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        # Create a second tensor of evenly spaced coordinates, shifted by 1
        coords_2 = (
            torch.arange(length_old, device=device, dtype=torch.float32) + 1
        ).view(1, 1, 1, -1)

        # Ensure the last coordinate doesn't go out of bounds
        coords_2[:, :, :, -1] -= 1

        # Interpolate the shifted coordinates to the new axis
        coords_2 = F.interpolate(
            coords_2, size=(1, length_new), mode="bilinear", align_corners=False
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # Linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # Reusing tensors for Linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new)

    # Expand the tensors to match the required dimensions
    coords_1 = coords_1.view(1, 1, -1, 1).expand((n, c, -1, w_new))
    coords_2 = coords_2.view(1, 1, -1, 1).expand((n, c, -1, w_new))
    ratios = ratios.view(1, 1, -1, 1).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result


def scale_latents(
    latents: Union[torch.Tensor, torch.FloatTensor],
    scale: float = 2.0,
    latent_scale_mode: LatentScaleModel = "bilinear",
    antialiased: bool = False,
):
    "Interpolate the latents to the desired scale."

<<<<<<< HEAD
    align_to = 32 if latent_scale_mode == "bislerp" else 8

=======
>>>>>>> origin/experimental
    s = time()

    logger.debug(f"Scaling latents with shape {list(latents.shape)}, scale: {scale}")

    # Scale and round to multiple of 32
    width_truncated = int(latents.shape[2] * scale)
    height_truncated = int(latents.shape[3] * scale)

    # Scale the latents
    if latent_scale_mode == "bislerp":
        interpolated = bislerp(latents, height_truncated, width_truncated)
    else:
        interpolated = F.interpolate(
            latents,
            size=(
                width_truncated,
                height_truncated,
            ),
            mode=latent_scale_mode,
            antialias=antialiased,
        )
    interpolated.to(latents.device, dtype=latents.dtype)
    logger.debug(f"Took {(time() - s):.2f}s to process latent upscale")
    logger.debug(f"Interpolated latents from {latents.shape} to {interpolated.shape}")
    return interpolated
