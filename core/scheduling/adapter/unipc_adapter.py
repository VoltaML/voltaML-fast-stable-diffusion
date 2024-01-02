import functools
from typing import Any, Callable, Optional, Union

import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from core.inference.utilities.philox import PhiloxGenerator

from ..types import Method, ModelType, SkipType, Variant
from ..unipc import NoiseScheduleVP, UniPC
from .k_adapter import KdiffusionSchedulerAdapter


class UnipcSchedulerAdapter(KdiffusionSchedulerAdapter):
    scheduler: NoiseScheduleVP

    skip_type: SkipType
    model_type: ModelType
    variant: Variant
    method: Method

    order: int
    lower_order_final: bool

    timesteps: torch.Tensor
    steps: int

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        skip_type: SkipType = "time_uniform",
        model_type: ModelType = "noise",
        variant: Variant = "bh1",
        method: Method = "multistep",
        order: int = 3,
        lower_order_final: bool = True,
    ) -> None:
        super().__init__(
            alphas_cumprod,
            "karras",
            ("", "", {}),
            (0, 0),
            0,
            False,
            0,
            0,
            (0, 0),
            0,
            device,
            dtype,
            sampler_settings={},
        )

        self.skip_type = skip_type
        self.model_type = model_type
        self.variant = variant
        self.method = method
        self.order = order
        self.lower_order_final = lower_order_final
        self.scheduler = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)

    def set_timesteps(
        self,
        steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.steps = steps

        def get_time_steps(skip_type, t_T, t_0, N, device):
            """Compute the intermediate time steps for sampling."""
            if skip_type == "logSNR":
                lambda_T = self.scheduler.marginal_lambda(torch.tensor(t_T).to(device))
                lambda_0 = self.scheduler.marginal_lambda(torch.tensor(t_0).to(device))
                logSNR_steps = torch.linspace(
                    lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1
                ).to(device)
                return self.scheduler.inverse_lambda(logSNR_steps)
            elif skip_type == "time_uniform":
                return torch.linspace(t_T, t_0, N + 1).to(device)
            elif skip_type == "time_quadratic":
                t_order = 2
                t = (
                    torch.linspace(
                        t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1
                    )
                    .pow(t_order)
                    .to(device)
                )
                return t
            else:
                raise ValueError(
                    f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'"
                )

        t_0 = 1.0 / self.scheduler.total_N
        t_T = self.scheduler.T
        self.timesteps = get_time_steps(
            skip_type=self.skip_type,
            t_T=t_T,
            t_0=t_0,
            N=self.steps,
            device=device or self.device,
        )

    def do_inference(
        self,
        x,
        call: Callable[..., Any],  # type: ignore
        apply_model: Callable[..., torch.Tensor],
        generator: Union[PhiloxGenerator, torch.Generator],
        callback,
        callback_steps,
        optional_device: Optional[torch.device] = None,
        optional_dtype: Optional[torch.dtype] = None,
        device: torch.device = None,  # type: ignore
    ) -> torch.Tensor:
        device = optional_device or call.device
        dtype = optional_dtype or call.dtype

        unet_or_controlnet = call

        def noise_pred_fn(x, t_continuous, cond=None, **model_kwargs):
            # Was originally get_model_input_time(t_continous)
            # but "schedule" is ALWAYS "discrete," so we can skip it :)
            t_input = (t_continuous - 1.0 / self.scheduler.total_N) * 1000
            if cond is None:
                output = unet_or_controlnet(
                    x.to(device=device, dtype=dtype),
                    t_input.to(device=device, dtype=dtype),
                    return_dict=False,
                    **model_kwargs,
                )
                if isinstance(unet_or_controlnet, UNet2DConditionModel):
                    output = output[0]
            else:
                output = unet_or_controlnet(
                    x.to(device=device, dtype=dtype),
                    t_input.to(device=device, dtype=dtype),
                    encoder_hidden_states=cond,
                    return_dict=False,
                    **model_kwargs,
                )
                if isinstance(unet_or_controlnet, UNet2DConditionModel):
                    output = output[0]
            return output

        def change_source(src):
            nonlocal unet_or_controlnet
            unet_or_controlnet = src

        apply_model = functools.partial(
            apply_model, call=noise_pred_fn, change_source=change_source
        )

        # predict_x0=True    ->   algorithm_type="data_prediction"
        # predict_x0=False   ->   algorithm_type="noise_prediction"
        uni_pc = UniPC(
            model_fn=apply_model,
            noise_schedule=self.scheduler,
            algorithm_type="data_prediction",
            variant=self.variant,
        )
        ret: torch.Tensor = uni_pc.sample(  # type: ignore
            x,
            timesteps=self.timesteps,
            steps=self.steps,
            method=self.method,
            order=self.order,
            lower_order_final=self.lower_order_final,
            callback=callback,
            callback_steps=callback_steps,
        )
        return ret.to(dtype=self.dtype, device=self.device)
