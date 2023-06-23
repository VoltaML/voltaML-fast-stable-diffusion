import logging
import math
from time import time
from typing import Optional, Union

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

from core.config import config
from core.flags import LatentScaleModel

logger = logging.getLogger(__name__)


def prepare_latents(
    pipe: StableDiffusionPipeline,
    image: Optional[torch.Tensor],
    timestep: torch.FloatTensor,
    batch_size: int,
    height: Optional[int],
    width: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator],
    latents=None,
):
    if image is None:
        shape = (
            batch_size,
            pipe.unet.config.in_channels,  # type: ignore
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )

        if latents is None:
            if device.type == "mps" or config.api.device_type == "directml":
                # randn does not work reproducibly on mps
                latents = torch.randn(
                    shape, generator=generator, device="cpu", dtype=dtype  # type: ignore
                ).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=generator.device, dtype=dtype  # type: ignore
                )
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
        if image.shape[1] != 4:
            init_latent_dist = pipe.vae.encode(image.to(config.api.device)).latent_dist  # type: ignore
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents
            init_latents = torch.cat([init_latents] * batch_size, dim=0)
        else:
            logger.debug("Skipping VAE encode, already have latents")
            init_latents = image

        init_latents_orig = init_latents
        shape = init_latents.shape

        # add noise to latents using the timesteps
        if device.type == "mps" or config.api.device_type == "directml":
            noise = torch.randn(
                shape, generator=generator, device="cpu", dtype=dtype
            ).to(device)
        else:
            # Retarded fix, but hey, if it works, it works
            if hasattr(pipe.vae, "main_device"):
                noise = torch.randn(
                    shape,
                    generator=torch.Generator("cpu").manual_seed(1),
                    device="cpu",
                    dtype=dtype,
                ).to(device)
            else:
                noise = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
        # Now this... I may have called the previous "hack" retarded, but this...
        # This just takes it to a whole new level
        latents = pipe.scheduler.add_noise(init_latents.to(device), noise.to(device), timestep.to(device))  # type: ignore
        return latents, init_latents_orig, noise


def bislerp_original(samples, width, height):
    shape = list(samples.shape)
    width_scale = (shape[3]) / (width)
    height_scale = (shape[2]) / (height)

    shape[3] = width
    shape[2] = height
    out1 = torch.empty(
        shape, dtype=samples.dtype, layout=samples.layout, device=samples.device
    )

    def algorithm(in1, in2, t):
        dims = in1.shape
        val = t

        # flatten to batches
        low = in1.reshape(dims[0], -1)
        high = in2.reshape(dims[0], -1)

        low_weight = torch.norm(low, dim=1, keepdim=True)
        low_weight[low_weight == 0] = 0.0000000001
        low_norm = low / low_weight
        high_weight = torch.norm(high, dim=1, keepdim=True)
        high_weight[high_weight == 0] = 0.0000000001
        high_norm = high / high_weight

        dot_prod = (low_norm * high_norm).sum(1)
        dot_prod[dot_prod > 0.9995] = 0.9995
        dot_prod[dot_prod < -0.9995] = -0.9995
        omega = torch.acos(dot_prod)
        so = torch.sin(omega)
        res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low_norm + (
            torch.sin(val * omega) / so
        ).unsqueeze(1) * high_norm
        res *= low_weight * (1.0 - val) + high_weight * val
        return res.reshape(dims)

    for x_dest in range(shape[3]):
        for y_dest in range(shape[2]):
            y = (y_dest + 0.5) * height_scale - 0.5
            x = (x_dest + 0.5) * width_scale - 0.5

            x1 = max(math.floor(x), 0)
            x2 = min(x1 + 1, samples.shape[3] - 1)
            wx = x - math.floor(x)

            y1 = max(math.floor(y), 0)
            y2 = min(y1 + 1, samples.shape[2] - 1)
            wy = y - math.floor(y)

            in1 = samples[:, :, y1, x1]
            in2 = samples[:, :, y1, x2]
            in3 = samples[:, :, y2, x1]
            in4 = samples[:, :, y2, x2]

            if (x1 == x2) and (y1 == y2):
                out_value = in1
            elif x1 == x2:
                out_value = algorithm(in1, in3, wy)
            elif y1 == y2:
                out_value = algorithm(in1, in2, wx)
            else:
                o1 = algorithm(in1, in2, wx)
                o2 = algorithm(in3, in4, wx)
                out_value = algorithm(o1, o2, wy)

            out1[:, :, y_dest, x_dest] = out_value
    return out1


def bislerp_gabeified(samples, width, height):
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

    s = time()

    logger.debug(f"Scaling latents with shape {list(latents.shape)}")

    # Scale and round to multiple of 32
    width_truncated = int(((latents.shape[2] * scale - 1) // 32 + 1) * 32)
    height_truncated = int(((latents.shape[3] * scale - 1) // 32 + 1) * 32)

    # Scale the latents
    if latent_scale_mode == "bislerp-tortured":
        interpolated = bislerp_gabeified(latents, height_truncated, width_truncated)
    elif latent_scale_mode == "bislerp-original":
        interpolated = bislerp_original(latents, height_truncated, width_truncated)
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
