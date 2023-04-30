import torch
import torch.nn.functional as F


def pred_x0(pipe, sample, model_output, timestep):
    """
    Modified from diffusers.schedulers.scheduling_ddim.DDIMScheduler.step
    Note: there are some schedulers that clip or do not return x_0 (PNDMScheduler, DDIMScheduler, etc.)
    """
    alpha_prod_t = pipe.scheduler.alphas_cumprod[
        timestep.to(pipe.scheduler.alphas_cumprod.device, dtype=torch.int64)
    ]

    beta_prod_t = 1 - alpha_prod_t
    if pipe.scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif pipe.scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        # predict V
        model_output = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {pipe.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
            " or `v_prediction`"
        )

    return pred_original_sample


def sag_masking(pipe, original_latents, attn_map, map_size, t, eps):
    "sag_masking"
    # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
    _, hw1, hw2 = attn_map.shape
    b, latent_channel, latent_h, latent_w = original_latents.shape
    h = pipe.unet.config.attention_head_dim
    if isinstance(h, list):
        h = h[-1]

    # Produce attention mask
    attn_map = attn_map.reshape(b, h, hw1, hw2)
    attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
    attn_mask = (
        attn_mask.reshape(b, map_size[0], map_size[1])
        .unsqueeze(1)
        .repeat(1, latent_channel, 1, 1)
        .type(attn_map.dtype)
    )
    attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

    # Blur according to the self-attention mask
    degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
    degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

    # Noise it again to match the noise level
    degraded_latents = pipe.scheduler.add_noise(
        degraded_latents, noise=eps, timesteps=torch.tensor([t])
    )

    return degraded_latents


def pred_epsilon(pipe, sample, model_output, timestep):
    "pred_epsilon"
    alpha_prod_t = pipe.scheduler.alphas_cumprod[
        timestep.to(pipe.scheduler.alphas_cumprod.device, dtype=torch.int64)
    ]

    beta_prod_t = 1 - alpha_prod_t
    if pipe.scheduler.config.prediction_type == "epsilon":
        pred_eps = model_output
    elif pipe.scheduler.config.prediction_type == "sample":
        pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (
            beta_prod_t**0.5
        )
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        pred_eps = (beta_prod_t**0.5) * sample + (alpha_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {pipe.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
            " or `v_prediction`"
        )

    return pred_eps


def gaussian_blur_2d(img, kernel_size, sigma):
    "Blurs an image with gaussian blur."
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
