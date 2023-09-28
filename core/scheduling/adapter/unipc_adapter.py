# pylint: disable=W0613

import functools
from typing import Any, Callable, Optional

import torch

from .k_adapter import KdiffusionSchedulerAdapter
from ..unipc import UniPC, NoiseScheduleVP
from ..types import Order, SkipType, ModelType, Method, Variant


class UnipcSchedulerAdapter(KdiffusionSchedulerAdapter):
    scheduler: NoiseScheduleVP

    skip_type: SkipType
    model_type: ModelType
    variant: Variant
    method: Method

    order: Order
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
        order: Order = 2,
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
        generator: torch.Generator,
        callback,
        callback_steps,
    ) -> torch.Tensor:
        def noise_pred_fn(x, t_continuous, cond=None, **model_kwargs):
            # Was originally get_model_input_time(t_continous)
            # but "schedule" is ALWAYS "discrete," so we can skip it :)
            t_input = (t_continuous - 1.0 / self.scheduler.total_N) * 1000
            if cond is None:
                output = call(
                    x.to(device=call.device, dtype=call.dtype),
                    t_input.to(device=call.device, dtype=call.dtype),
                    return_dict=True,
                    **model_kwargs,
                )[0]
            else:
                output = call(x.to(device=call.device, dtype=call.dtype), t_input.to(device=call.device, dtype=call.dtype), return_dict=True, encoder_hidden_states=cond, **model_kwargs)[0]  # type: ignore
            if self.model_type == "noise":
                return output
            elif self.model_type == "x_start":
                alpha_t, sigma_t = self.scheduler.marginal_alpha(
                    t_continuous
                ), self.scheduler.marginal_std(t_continuous)
                return (x - alpha_t * output) / sigma_t
            elif self.model_type == "v":
                alpha_t, sigma_t = self.scheduler.marginal_alpha(
                    t_continuous
                ), self.scheduler.marginal_std(t_continuous)
                return alpha_t * output + sigma_t * x
            elif self.model_type == "score":
                sigma_t = self.scheduler.marginal_std(t_continuous)
                return -sigma_t * output

        apply_model = functools.partial(apply_model, call=noise_pred_fn)

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
