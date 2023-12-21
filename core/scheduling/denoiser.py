import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser

from .types import Denoiser


class _ModelWrapper:
    def __init__(self, alphas_cumprod: torch.Tensor) -> None:
        self.callable: torch.Module = None  # type: ignore
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs) -> torch.Tensor:
        "denoiser#apply_model"
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        if isinstance(self.callable, torch.nn.Module):
            ret = self.callable(*args, encoder_hidden_states=encoder_hidden_states, return_dict=False, **kwargs)  # type: ignore
            if isinstance(self.callable, UNet2DConditionModel):
                return ret[0]
            return ret
        else:
            return self.callable(*args, encoder_hidden_states=encoder_hidden_states, **kwargs)  # type: ignore


def create_denoiser(
    alphas_cumprod: torch.Tensor,
    prediction_type: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    denoiser_enable_quantization: bool = False,
) -> Denoiser:
    "Create a denoiser based on the provided prediction_type"
    model = _ModelWrapper(alphas_cumprod)
    model.alphas_cumprod = alphas_cumprod
    if prediction_type == "v_prediction":
        denoiser = CompVisVDenoiser(model, quantize=denoiser_enable_quantization)
    else:
        denoiser = CompVisDenoiser(model, quantize=denoiser_enable_quantization)
    denoiser.sigmas = denoiser.sigmas.to(device, dtype)
    denoiser.log_sigmas = denoiser.log_sigmas.to(device, dtype)
    return denoiser
