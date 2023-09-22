from typing import Union

import torch
from kdiffusion.external import CompVisVDenoiser, CompVisDenoiser

Denoiser = Union[CompVisDenoiser, CompVisVDenoiser]

class _ModelWrapper:
    def __init__(self, model: torch.Module, alphas_cumprod) -> None:
        self.model: torch.Module = model
        self.alphas_cumprod = alphas_cumprod
    
    def apply_model(self, *args, **kwargs) -> None:
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[-2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample  # type: ignore

def create_denoiser(
        unet: torch.Module,
        alphas_cumprod,
        prediction_type: str,
        device: torch.device,
        dtype: torch.dtype,
        denoiser_enable_quantization: bool = False
) -> Denoiser:
    model = _ModelWrapper(unet, alphas_cumprod)
    if prediction_type == "v_prediction":
        denoiser = CompVisVDenoiser(model, quantize=denoiser_enable_quantization)
    else:
        denoiser = CompVisDenoiser(model, quantize=denoiser_enable_quantization)
    denoiser.sigmas = denoiser.sigmas.to(device, dtype)
    denoiser.log_sigmas = denoiser.log_sigmas.to(device, dtype)
    return denoiser