# pylint: disable=unused-argument

import functools
import inspect
from typing import Callable, Optional, Tuple, Union

import k_diffusion
import torch

from core.config import config
from core.inference.utilities.philox import PhiloxGenerator

from ..hijack import TorchHijack
from ..sigmas import build_sigmas
from ..types import Denoiser, Sampler, SigmaScheduler

sampling = k_diffusion.sampling


class KdiffusionSchedulerAdapter:
    "Somewhat diffusers compatible scheduler-like K-diffusion adapter."
    sampler_tuple: Sampler
    denoiser: Denoiser

    # diffusers compat
    config: dict = {"steps_offset": 0, "prediction_type": "epsilon"}

    # should really be "sigmas," but for compatibility with diffusers
    # it's named timesteps.
    timesteps: torch.Tensor  # calculated sigmas
    scheduler_name: Optional[SigmaScheduler]

    alphas_cumprod: torch.Tensor

    sigma_range: Tuple[float, float] = (0, 1.0)
    sigma_rho: float = 1
    sigma_always_discard_next_to_last: bool = False

    sampler_eta: Optional[float] = None
    sampler_churn: Optional[float] = None
    sampler_trange: Tuple[float, float] = (0, 0)
    sampler_noise: Optional[float] = None

    steps: int = 50

    eta_noise_seed_delta: float = 0

    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        scheduler_name: Optional[SigmaScheduler],
        sampler_tuple: Sampler,
        sigma_range: Tuple[float, float],
        sigma_rho: float,
        sigma_discard: bool,
        sampler_eta: float,
        sampler_churn: float,
        sampler_trange: Tuple[float, float],
        sampler_noise: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.alphas_cumprod = alphas_cumprod.to(device=device)

        self.scheduler_name = scheduler_name
        self.sampler_tuple = sampler_tuple

        self.sigma_range = sigma_range
        self.sigma_rho = sigma_rho
        self.sigma_always_discard_next_to_last = sigma_discard

        self.sampler_eta = sampler_eta
        if self.sampler_eta is None:
            self.sampler_eta = self.sampler_tuple[2].get("default_eta", None)
        self.sampler_churn = sampler_churn
        self.sampler_trange = sampler_trange
        self.sampler_noise = sampler_noise

        self.device = device
        self.dtype = dtype

    def set_timesteps(
        self,
        steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        "Initialize timesteps (sigmas) and set steps to correct amount."
        self.steps = steps
        self.timesteps = build_sigmas(
            steps=steps,
            denoiser=self.denoiser,
            discard_next_to_last_sigma=self.sampler_tuple[2].get(
                "discard_next_to_last_sigma", self.sigma_always_discard_next_to_last
            ),
            scheduler=self.scheduler_name,
            custom_rho=self.sigma_rho,
            custom_sigma_min=self.sigma_range[0],
            custom_sigma_max=self.sigma_range[1],
        ).to(device=device or self.device, dtype=dtype or self.dtype)

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Optional[int] = None
    ) -> torch.FloatTensor:
        "diffusers#scale_model_input"
        return sample

    @property
    def init_noise_sigma(self) -> torch.Tensor:
        "diffusers#init_noise_sigma"
        # SGM / ODE doesn't necessarily produce "better" images, it's here for feature parity with both A1111 and SGM.
        return (
            torch.sqrt(1.0 + self.timesteps[0] ** 2.0)
            if config.api.sgm_noise_multiplier
            else self.timesteps[0]
        )

    def do_inference(
        self,
        x: torch.Tensor,
        call: Callable,
        apply_model: Callable[..., torch.Tensor],
        generator: Union[PhiloxGenerator, torch.Generator],
        callback,
        callback_steps,
    ) -> torch.Tensor:
        "Run inference function provided with denoiser."
        apply_model = functools.partial(apply_model, call=self.denoiser)
        self.denoiser.inner_model.callable = call

        def callback_func(data):
            if callback is not None and data["i"] % callback_steps == 0:
                callback(data["i"], data["sigma"], data["x"])

        def create_noise_sampler():
            if self.sampler_tuple[2].get("brownian_noise", False):
                return k_diffusion.sampling.BrownianTreeNoiseSampler(
                    x,
                    self.timesteps[self.timesteps > 0].min(),
                    self.timesteps.max(),
                )

            def noiser(sigma=None, sigma_next=None):
                from core.inference.utilities import randn_like

                return randn_like(x, generator, device=x.device, dtype=x.dtype)

            return noiser

        sampler_args = {
            "n": self.steps,
            "model": apply_model,
            "x": x,
            "callback": callback_func,
            "sigmas": self.timesteps,
            "sigma_min": self.denoiser.sigmas[0].item(),  # type: ignore
            "sigma_max": self.denoiser.sigmas[-1].item(),  # type: ignore
            "noise_sampler": create_noise_sampler(),
            "eta": self.sampler_eta,
            "s_churn": self.sampler_churn,
            "s_tmin": self.sampler_trange[0],
            "s_tmax": self.sampler_trange[1],
            "s_noise": self.sampler_noise,
            "order": 2 if self.sampler_tuple[2].get("second_order", False) else None,
        }

        k_diffusion.sampling.torch = TorchHijack()

        if isinstance(self.sampler_tuple[1], str):
            sampler_func = getattr(sampling, self.sampler_tuple[1])
        else:
            sampler_func = self.sampler_tuple[1]
        parameters = inspect.signature(sampler_func).parameters.keys()
        for key, value in sampler_args.copy().items():
            if key not in parameters or value is None:
                del sampler_args[key]
        return sampler_func(**sampler_args)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        "diffusers#add_noise"
        return original_samples + (noise * self.init_noise_sigma).to(
            original_samples.device, original_samples.dtype
        )
