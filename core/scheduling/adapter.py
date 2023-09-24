from typing import Callable, Optional, Tuple
import inspect
import functools

import k_diffusion
import torch
from tqdm import tqdm

from .sigmas import build_sigmas
from .denoiser import Denoiser, _ModelWrapper

sampling = k_diffusion.sampling

class KdiffusionSchedulerAdapter:
    sampler_tuple: tuple[None, str, dict]  # selected sampler
    denoiser: Denoiser  # selected denoiser

    # diffusers compat
    config: dict = {"steps_offset": 0}

    # should really be "sigmas," but for compatibility with diffusers
    # it's named timesteps.
    timesteps: torch.Tensor  # calculated sigmas
    scheduler_name: str  # name of the scheduler (karras or polyexponential)

    alphas_cumprod: torch.Tensor


    sigma_range: Tuple[float, float] = (0, 1.0)
    sigma_rho: float = 1
    sigma_always_discard_next_to_last: bool = False

    sampler_eta: Optional[float] = None
    sampler_churn: Optional[float] = None
    sampler_t: Tuple[float, float] = (0, 0)
    sampler_noise: Optional[float] = None

    steps: int = 50

    eta_noise_seed_delta: float = 0

    def __init__(self, alphas_cumprod, scheduler_name, sampler_tuple, sigma_range, sigma_rho, sigma_discard, sampler_eta, sampler_churn, sampler_tmin, sampler_tmax, sampler_noise) -> None:
        self.alphas_cumprod = alphas_cumprod

        self.scheduler_name = scheduler_name
        self.sampler_tuple = sampler_tuple

        # SAG compat.
        if scheduler_name == "polyexponential":
            self.config.prediction_type = "epsilon"  # type: ignore
        else:
            self.config.prediction_type = "v_prediction"  # type: ignore
        self.sigma_range = sigma_range
        self.sigma_rho = sigma_rho
        self.sigma_always_discard_next_to_last = sigma_discard

        self.sampler_eta = sampler_eta
        if self.sampler_eta == None:
            self.sampler_eta = self.sampler_tuple[2].get("default_eta", None)
        self.sampler_churn = sampler_churn
        self.sampler_t = (sampler_tmin, sampler_tmax)
        self.sampler_noise = sampler_noise

    def set_timesteps(self, steps: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.steps = steps
        self.timesteps = build_sigmas(
            # Not exactly multiplying steps by 2, but this'll do for now...
            steps=steps * (2 if self.sampler_tuple[2].get("second_order", False) else 1),
            denoiser=self.denoiser,
            discard_next_to_last_sigma=self.sampler_tuple[2].get("discard_next_to_last_sigma", self.sigma_always_discard_next_to_last),
            scheduler=self.scheduler_name,

            custom_rho=self.sigma_rho,
            custom_sigma_min=self.sigma_range[0],
            custom_sigma_max=self.sigma_range[1]
        ).to(device=device, dtype=dtype)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        return sample

    def do_inference(
        self,
        x,
        apply_model: Callable[..., torch.Tensor],
        progress_bar: tqdm,
        generator,
        callback,
        callback_steps,
    ) -> Callable:
        if progress_bar is not None:
            apply_model = functools.partial(apply_model, tqdm=progress_bar)

        self.denoiser.inner_model.model = apply_model

        def sampler_call(x, t):
            return self.denoiser(x, t)

        def callback_func(data):
            if callback is not None and data["i"] % callback_steps == 0:
                callback(data["i"], data["sigma"], data["x"])

        def create_noise_sampler():
            if self.sampler_tuple[2].get("brownian_noise", False):
                return k_diffusion.sampling.BrownianTreeNoiseSampler(x, self.timesteps[self.timesteps > 0].min(), self.timesteps.max(), seed=[generator.seed])

        sampler_args = {
            "n": self.steps,
            "model": sampler_call,
            "x": x,
            "callback": callback_func,
            "sigmas": self.timesteps,
            "sigma_min": self.denoiser.sigmas[0].item(),  # type: ignore
            "sigma_max": self.denoiser.sigmas[-1].item(),  # type: ignore
            "noise_sampler": create_noise_sampler(),

            "eta": self.sampler_eta,
            "s_churn": self.sampler_churn,
            "s_tmin": self.sampler_t[0],
            "s_tmax": self.sampler_t[1],
            "s_noise": self.sampler_noise
        }

        sampler_func = getattr(sampling, self.sampler_tuple[1])
        parameters = inspect.signature(sampler_func)
        for key in sampler_args.copy().keys():
            if key not in parameters or sampler_args[key] is None:  # type: ignore
                del sampler_args[key]
        return sampler_func(**sampler_args)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
