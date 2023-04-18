from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

from core.optimizations import send_to_gpu


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
            if device.type == "mps":
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
        if hasattr(pipe.vae, "main_device"):  # type: ignore
            send_to_gpu(pipe.vae, None)  # type: ignore
        init_latent_dist = pipe.vae.encode(image).latent_dist  # type: ignore
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents
        init_latents = torch.cat([init_latents] * batch_size, dim=0)
        init_latents_orig = init_latents
        shape = init_latents.shape

        # add noise to latents using the timesteps
        if device.type == "mps":
            noise = torch.randn(
                shape, generator=generator, device="cpu", dtype=dtype
            ).to(device)
        else:
            noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = pipe.scheduler.add_noise(init_latents, noise, timestep)  # type: ignore
        return latents, init_latents_orig, noise
