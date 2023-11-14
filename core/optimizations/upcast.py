import logging

import torch
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from diffusers.models.autoencoder_kl import AutoencoderKL

from core.config import config


logger = logging.getLogger(__name__)


def upcast_vae(vae: AutoencoderKL, sample: torch.Tensor):
    if (
        vae.config["force_upcast"] or config.api.upcast_vae
    ) and vae.dtype == torch.float16:
        dtype = vae.dtype
        logger.info(
            'Upcasting VAE to FP32 (vae["force_upcast"] OR config.api.upcast_vae)'
        )
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
            vae.post_quant_conv.to(dtype=dtype)
            vae.decoder.conv_in.to(dtype=dtype)
            vae.decoder.mid_block.to(dtype=dtype)  # type: ignore
    samp = sample.to(dtype=next(iter(vae.post_quant_conv.parameters())).dtype)
    return vae.decode(
        samp,  # type: ignore
        return_dict=False,
    )[0]
