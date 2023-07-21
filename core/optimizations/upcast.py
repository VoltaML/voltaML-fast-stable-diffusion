from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
import torch


def upcast_vae(vae: AutoencoderKL):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,  # type: ignore
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)  # type: ignore
