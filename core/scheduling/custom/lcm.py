from k_diffusion.sampling import default_noise_sampler
import torch
from tqdm import trange


@torch.no_grad()
def sample_lcm(
    model,
    x: torch.Tensor,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x
