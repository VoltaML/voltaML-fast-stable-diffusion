# Taken from https://github.com/wl-zhao/UniPC/blob/main/uni_pc.py#L236

import logging
from typing import Callable, List, Optional, Tuple

import torch
from tqdm import tqdm

from ..types import AlgorithmType, Method, SkipType, Variant
from .noise_scheduler import NoiseScheduleVP
from .utility import expand_dims

logger = logging.getLogger(__name__)


class UniPC:
    "https://github.com/wl-zhao/UniPC/blob/main/uni_pc.py#L236"
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    noise_schedule: NoiseScheduleVP

    correcting_x0_fn: Optional[
        Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    ]
    correcting_xt_fn: Optional[Callable]

    dynamic_thresholding_ratio: float
    thresholding_max_val: float

    variant: Variant
    predict_x0: bool

    def __init__(
        self,
        model_fn: Callable,
        noise_schedule: NoiseScheduleVP,
        algorithm_type: AlgorithmType = "data_prediction",
        correcting_x0_fn: Optional[Callable] = None,
        correcting_xt_fn: Optional[Callable] = None,
        thresholding_max_val: float = 1.0,
        dynamic_thresholding_ratio: float = 0.995,
        variant: Variant = "bh1",
    ) -> None:
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """

        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.variant = variant
        self.predict_x0 = algorithm_type == "data_prediction"

    def dynamic_thresholding_fn(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s).to(s.device)
            ),
            dims,
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(
            t
        ), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, None)
        return x0

    def model_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(
        self,
        skip_type: SkipType,
        t_T: int,
        t_0: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(
                lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1
            ).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1)
                .pow(t_order)
                .to(device)
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self,
        steps: int,
        order: int,
        skip_type: SkipType,
        t_T: int,
        t_0: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(t.shape) == 0:
            t = t.view(-1)
        if "bh" in self.variant:
            return self.multistep_uni_pc_bh_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )
        else:
            assert self.variant == "vary_coeff"
            return self.multistep_uni_pc_vary_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )

    def multistep_uni_pc_vary_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
        use_corrector: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(
            f"using unified predictor-corrector with order {order} (solver type: vary coeff)"
        )
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            logger.debug("using corrector")
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= k + 1

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]  # type: ignore
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]  # type: ignore
                    )
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])  # type: ignore
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
                t_prev_0
            ), ns.marginal_log_mean_coeff(t)
            x_t_ = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (
                sigma_t * h_phi_1
            ) * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]  # type: ignore
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]  # type: ignore
                    )
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])  # type: ignore
        return x_t, model_t  # type: ignore

    def multistep_uni_pc_bh_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
        x_t: Optional[torch.Tensor] = None,
        use_corrector: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(
            f"using unified predictor-corrector with order {order} (solver type: B(h))"
        )
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0
        ), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            logger.debug("using corrector")
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)  # type: ignore
                else:
                    pred_res = 0
                x_t = x_t_ - alpha_t * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)  # type: ignore
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)  # type: ignore
        else:
            x_t_ = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - sigma_t * h_phi_1 * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)  # type: ignore
                else:
                    pred_res = 0
                x_t = x_t_ - sigma_t * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)  # type: ignore
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)  # type: ignore
        return x_t, model_t  # type: ignore

    def sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        steps: int = 20,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        order: int = 2,
        method: Method = "multistep",
        lower_order_final: bool = True,
        denoise_to_zero: bool = False,
        callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Compute the sample at time `t_end` by UniPC, given the initial `x` at time `t_start`.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []

        progress_bar = tqdm(total=steps)

        with torch.no_grad():
            if method == "multistep":
                assert steps >= order
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                # Init the first `order` values by lower order multistep UniPC.
                for step in range(1, order):
                    t = timesteps[step]
                    x, model_x = self.multistep_uni_pc_update(
                        x, model_prev_list, t_prev_list, t, step, use_corrector=True  # type: ignore
                    )
                    if model_x is None:
                        model_x = self.model_fn(x, t)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(model_x)
                    if callback is not None and step % callback_steps == 0:
                        callback(step, t, x)
                    progress_bar.update()

                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        logger.debug("do not run corrector at the last step")
                        use_corrector = False
                    else:
                        use_corrector = True
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        t,
                        step_order,  # type: ignore
                        use_corrector=use_corrector,
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        if model_x is None:
                            model_x = self.model_fn(x, t)
                        model_prev_list[-1] = model_x
                    if callback is not None and step % callback_steps == 0:
                        callback(step, t, x)
                    progress_bar.update()
            else:
                raise ValueError(f"Got wrong method {method}")

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        progress_bar.close()
        if return_intermediate:
            return x, intermediates  # type: ignore
        else:
            return x
