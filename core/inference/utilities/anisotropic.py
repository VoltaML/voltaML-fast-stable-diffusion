# Taken from lllyasviel/Fooocus
# Show some love over at https://github.com/lllyasviel/Fooocus/

from typing import Union, Tuple, Optional

import torch


def _compute_zero_padding(kernel_size: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def gaussian(
    window_size: int,
    sigma: Union[torch.Tensor, float],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    batch_size = sigma.shape[0]  # type: ignore

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)  # type: ignore
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))  # type: ignore

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel2d(
    kernel_size: Union[Tuple[int, int], int],
    sigma: Union[Tuple[float, float], torch.Tensor],
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)  # type: ignore

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(
        ksize_y, sigma_y, force_even, device=device, dtype=dtype
    )[..., None]
    kernel_x = get_gaussian_kernel1d(
        ksize_x, sigma_x, force_even, device=device, dtype=dtype
    )[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: Union[Tuple[int, int], int],
    sigma: Union[Tuple[float, float], torch.Tensor],
) -> torch.Tensor:
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=input.device, dtype=input.dtype)
    else:
        sigma = sigma.to(device=input.device, dtype=input.dtype)

    ky, kx = _unpack_2d_ks(kernel_size)
    bs = sigma.shape[0]
    kernel_x = get_gaussian_kernel1d(kx, sigma[:, 1].view(bs, 1))
    kernel_y = get_gaussian_kernel1d(ky, sigma[:, 0].view(bs, 1))
    out = filter2d_separable(input, kernel_x, kernel_y)

    return out


def filter2d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
) -> torch.Tensor:
    out_x = filter2d(input, kernel_x[..., None, :])
    out = filter2d(out_x, kernel_y[..., None])
    return out


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )
    out = output.view(b, c, h, w)

    return out


def unsharp_mask(
    input: torch.Tensor,
    kernel_size: Union[Tuple[int, int], int],
    sigma: Union[Tuple[float, float], torch.Tensor],
) -> torch.Tensor:
    data_blur: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)
    data_sharpened: torch.Tensor = input + (input - data_blur)
    return data_sharpened


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: Union[float, torch.Tensor],
    force_even: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def _compute_padding(kernel_size: list[int]) -> list[int]:
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _bilateral_blur(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    kernel_size: Union[Tuple[int, int], int],
    sigma_color: Union[float, torch.Tensor],
    sigma_space: Union[Tuple[float, float], torch.Tensor],
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    if isinstance(sigma_color, torch.Tensor):
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(
            -1, 1, 1, 1, 1
        )

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = torch.nn.functional.pad(
        input, (pad_x, pad_x, pad_y, pad_y), mode=border_type
    )
    unfolded_input = (
        padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
    )  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = torch.nn.functional.pad(
            guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type
        )
        unfolded_guidance = (
            padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
        )  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (
        -0.5 / sigma_color**2 * color_distance_sq
    ).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(
        kernel_size, sigma_space, device=input.device, dtype=input.dtype  # type: ignore
    )
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def adaptive_anisotropic_filter(x, g=None):
    if g is None:
        g = x
    s, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)
    s = s + 1e-5
    guidance = (g - m) / s
    y = _bilateral_blur(
        x,
        guidance,
        kernel_size=(13, 13),
        sigma_color=3.0,
        sigma_space=3.0,  # type: ignore
        border_type="reflect",
        color_distance_type="l1",
    )
    return y
