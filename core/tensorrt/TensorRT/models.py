# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import torch
from transformers import CLIPTextModel

from core.submodules.diffusers.src.diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)

from .. import models


class CLIP(models.CLIP):
    "CLIP model for text to image generation (takes words and creates input latent)"

    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token=hf_token,
            text_maxlen=text_maxlen,
            embedding_dim=embedding_dim,
            fp16=fp16,
            device=device,
            verbose=verbose,
            max_batch_size=max_batch_size,
        )
        self.model_id = model_id

    def get_model(self):
        "Return the loaded model"

        model = CLIPTextModel.from_pretrained(self.model_id)
        assert isinstance(model, CLIPTextModel)
        model.to(self.device)
        return


class UNet(models.UNet):
    "UNet model for image generation (makes a better prediction than the last step over and over again)"

    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token=hf_token,
            text_maxlen=text_maxlen,
            embedding_dim=embedding_dim,
            fp16=fp16,
            device=device,
            verbose=verbose,
            max_batch_size=max_batch_size,
        )
        self.model_id = model_id

    def get_model(self):
        "Return the loaded model"

        model_opts = (
            {"revision": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        )
        model = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", use_auth_token=self.hf_token, **model_opts
        )

        assert isinstance(model, UNet2DConditionModel)
        model.to(self.device)
        return model


class VAE(models.VAE):
    "VAE model for image generation (takes the last step and upscales it to 8x the size)"

    def __init__(
        self,
        model_id: str,
        hf_token="",
        text_maxlen=77,
        embedding_dim=768,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=16,
    ):
        super().__init__(
            hf_token=hf_token,
            text_maxlen=text_maxlen,
            embedding_dim=embedding_dim,
            fp16=fp16,
            device=device,
            verbose=verbose,
            max_batch_size=max_batch_size,
        )
        self.model_id = model_id

    def get_model(self):
        "Return the loaded model"

        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            use_auth_token=self.hf_token,
        )
        assert isinstance(vae, AutoencoderKL)
        vae.to(self.device)
        vae.forward = vae.decode  # type: ignore
        return vae
