import torch

import k_diffusion
from .anisotropic import adaptive_anisotropic_filter, unsharp_mask
from core.config import config

cfg_x0, cfg_s, cfg_cin, eps_record = None, None, None, None


def patched_ddpm_denoiser_forward(self, input, sigma, **kwargs):
    global cfg_x0, cfg_s, cfg_cin, eps_record

    c_out, c_in = [
        k_diffusion.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)
    ]
    cfg_x0, cfg_s, cfg_cin = input, c_out, c_in
    eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
    if eps_record is not None:
        eps_record = eps.clone().cpu()
    return input + eps * c_out


def patched_vddpm_denoiser_forward(self, input, sigma, **kwargs):
    global cfg_x0, cfg_s, cfg_cin

    c_skip, c_out, c_in = [
        k_diffusion.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)
    ]
    cfg_x0, cfg_s, cfg_cin = input, c_out, c_in
    return (
        self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out
        + input * c_skip
    )


k_diffusion.external.DiscreteEpsDDPMDenoiser.forward = patched_ddpm_denoiser_forward
k_diffusion.external.DiscreteVDDPMDenoiser.forward = patched_vddpm_denoiser_forward


def calculate_cfg(
    cond: torch.Tensor, uncond: torch.Tensor, cfg: float, timestep: torch.IntTensor
):
    if config.api.apply_unsharp_mask:
        cc = uncond + cfg * (cond - uncond)

        MIX_FACTOR = 0.003
        cond_scale_factor = min(0.02 * cfg, 0.65)
        usm_sigma = torch.clamp(1 + timestep * cond_scale_factor, min=1e-6)
        sharpened = unsharp_mask(cond, (3, 3), (usm_sigma, usm_sigma), border_type="reflect")  # type: ignore

        return cc + (sharpened - cc) * MIX_FACTOR

    if config.api.cfg_rescale_threshold == "off":
        return uncond + cfg * (cond - uncond)

    if config.api.cfg_rescale_threshold <= cfg:
        global cfg_x0, cfg_s

        if cfg_x0.shape[0] == 2:  # type: ignore
            cfg_x0, _ = cfg_x0.chunk(2)  # type: ignore

        positive_x0 = cond * cfg_s + cfg_x0
        t = 1.0 - (timestep / 999.0)[:, None, None, None].clone()
        # Magic number: 2.0 is "sharpness"
        alpha = 0.001 * 2.0 * t

        positive_eps_degraded = adaptive_anisotropic_filter(x=cond, g=positive_x0)
        cond = positive_eps_degraded * alpha + cond * (1.0 - alpha)

        reps = (uncond + cfg * (cond - uncond)) * t
        # Magic number: 0.7 is "base cfg"
        mimicked = (uncond + 0.7 * (cond - uncond)) * (1 - t)
        return reps + mimicked

    return uncond + cfg * (cond - uncond)
