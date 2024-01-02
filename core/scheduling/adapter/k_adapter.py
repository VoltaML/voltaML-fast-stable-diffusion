import functools
import inspect
import logging
from typing import Callable, Optional, Tuple, Union

import k_diffusion
import torch

from core.config import config
from core.inference.utilities.philox import PhiloxGenerator
from core.types import SigmaScheduler

from ..hijack import TorchHijack
from ..sigmas import build_sigmas
from ..types import Denoiser, Sampler

sampling = k_diffusion.sampling
logger = logging.getLogger(__name__)


class KdiffusionSchedulerAdapter:
    "Somewhat diffusers compatible scheduler-like K-diffusion adapter."
    sampler_tuple: Sampler
    denoiser: Denoiser

    # diffusers compat
    config: dict = {
        "steps_offset": 0,
        "prediction_type": "epsilon",
        "num_train_timesteps": 1000,
    }

    # should really be "sigmas," but for compatibility with diffusers
    # it's named timesteps.
    timesteps: torch.Tensor  # calculated sigmas
    scheduler_name: Optional[SigmaScheduler]

    alphas_cumprod: torch.Tensor

    sigma_range: Tuple[float, float] = (0, 1.0)
    sigma_rho: Optional[float] = None
    sigma_always_discard_next_to_last: bool = False

    sampler_eta: Optional[float] = None
    sampler_churn: Optional[float] = None
    sampler_trange: Tuple[float, float] = (0, 0)
    sampler_noise: Optional[float] = None

    steps: int = 50

    eta_noise_seed_delta: float = 0

    device: torch.device
    dtype: torch.dtype

    sampler_settings: dict

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
        sampler_settings: dict,
    ) -> None:
        self.alphas_cumprod = alphas_cumprod.to(device=device)

        self.scheduler_name = scheduler_name
        self.sampler_tuple = sampler_tuple

        self.sigma_range = sigma_range

        self.sampler_settings = sampler_settings
        if self.sampler_settings:
            for key, value in self.sampler_settings.copy().items():
                if value is None:
                    del self.sampler_settings[key]

            logger.debug(f"Sampler settings overwrite: {self.sampler_settings}")

        self.sampler_eta = sampler_eta
        if self.sampler_eta is None:
            self.sampler_eta = self.sampler_tuple[2].get("default_eta", None)
        self.sampler_churn = sampler_churn
        self.sampler_trange = sampler_trange
        self.sampler_noise = sampler_noise

        self.device = device
        self.dtype = dtype

        self.sigma_rho = sigma_rho
        self.sigma_always_discard_next_to_last = sigma_discard

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
        apply_model: Callable[
            [
                torch.Tensor,
                torch.IntTensor,
                Callable[..., torch.Tensor],
                Callable[[Callable], None],
            ],
            torch.Tensor,
        ],
        generator: Union[PhiloxGenerator, torch.Generator],
        callback,
        callback_steps,
        device: torch.device = None,  # type: ignore
    ) -> torch.Tensor:
        "Run inference function provided with denoiser."

        def change_source(src):
            self.denoiser.inner_model.callable = src

        apply_model = functools.partial(apply_model, call=self.denoiser, change_source=change_source)  # type: ignore
        change_source(call)

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
            "sigmas": self.timesteps.to(device=x.device),
            "sigma_min": self.denoiser.sigmas[0].item(),  # type: ignore
            "sigma_max": self.denoiser.sigmas[-1].item(),  # type: ignore
            "noise_sampler": create_noise_sampler(),
            "eta": self.sampler_eta,
            "s_churn": self.sampler_churn,
            "s_tmin": self.sampler_trange[0],
            "s_tmax": self.sampler_trange[1],
            "s_noise": self.sampler_noise,
            "order": 2 if self.sampler_tuple[2].get("second_order", False) else None,
            **self.sampler_settings,
        }

        k_diffusion.sampling.torch = TorchHijack(generator)

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
