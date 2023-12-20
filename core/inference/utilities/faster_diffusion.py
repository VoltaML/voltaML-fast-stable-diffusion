from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
import torch


def multistep_pre(
    scheduler: LMSDiscreteScheduler, noise_pred: torch.Tensor, t: list, x: torch.Tensor
):
    step_span = len(t)
    batch_size = noise_pred.shape[0]
    batch_per_step = batch_size // step_span

    denoised = x
    for i, timestep in enumerate(t):
        curr_noise = noise_pred[i * batch_per_step : (i + 1) * batch_per_step]
        denoised = scheduler.step(curr_noise, timestep, denoised).prev_sample  # type: ignore
    return denoised


def warp_feature(sample: torch.Tensor, step: int):
    uncond, cond = sample.chunk(2)
    if uncond.dim() == 5:
        uncond = uncond.repeat(step, 1, 1, 1, 1)
        cond = cond.repeat(step, 1, 1, 1, 1)
    else:
        uncond = uncond.repeat(step, 1, 1, 1)
        cond = cond.repeat(step, 1, 1, 1)
    return torch.cat([uncond, cond])


def warp_text(sample: torch.Tensor, step: int):
    uncond, cond = sample.chunk(2)
    uncond = uncond.repeat(step, 1, 1)
    cond = cond.repeat(step, 1, 1)
    return torch.cat([uncond, cond])


def warp_controlnet_block_samples(block_samples, step):
    if block_samples is None:
        return block_samples
    output = []
    for sample in block_samples:
        output.append(warp_feature(sample, step))
    return tuple(output)


def warp_timestep(timestep: torch.Tensor, batch_size):
    batch_size = batch_size // 2
    output = []
    for t in timestep:
        output.append(t[None].expand(batch_size))
    out = torch.cat(output).repeat(2, 1).reshape(-1)
    return out


_times = [
    0,
    1,
    2,
    3,
    5,
    10,
    15,
    25,
    35,
    50,
    65,
    85,
    100,
    125,
    155,
    185,
    220,
    255,
    295,
    345,
    390,
    435,
    485,
    535,
    590,
    645,
]


def get_span(idx: int, total_steps: int) -> tuple[int, int]:
    idx = max(idx, 1)  # make sure idx is at least 1 so we don't get 0-tensors.
    adjusted_times = list(
        map(lambda x: _times[x] - _times[max(0, x - 1)], range(len(_times)))
    )
    sum_steps = sum(adjusted_times[:idx])
    if sum_steps > total_steps:
        return max(total_steps - sum_steps, 0), total_steps
    return _times[idx] - _times[idx - 1], sum_steps


def adjust_steps_to_idx(total_steps: int) -> int:
    idx = 0
    sum_steps = 0
    while total_steps - sum_steps > 0:
        idx += 1
        _, sum_steps = get_span(idx, total_steps)
    return idx - 1
