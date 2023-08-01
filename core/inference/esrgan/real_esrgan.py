import os
from typing import Optional

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from core.config import config
from core.inference.base_model import InferenceModel
from core.types import Job, UpscaleQueueEntry
from core.utils import convert_to_image


class RealESRGAN(InferenceModel):
    "High level model wrapper for RealESRGAN models"

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        scale: int = 4,
        denoise_strength: float = 1.0,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
    ):
        super().__init__(model_id=model_name, device=device)

        self.denoise_strength = denoise_strength
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.scale = scale

        self.upsampler: Optional[RealESRGANer] = None

        self.load()

    @property
    def gpu_id(self) -> int:
        "Returns the GPU ID"

        return config.api.device_id

    def load(self):
        if self.model_id == "RealESRGAN_x4plus":  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ]
        elif self.model_id == "RealESRNet_x4plus":  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            ]
        elif (
            self.model_id == "RealESRGAN_x4plus_anime_6B"
        ):  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ]
        elif self.model_id == "RealESRGAN_x2plus":  # x2 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            netscale = 2
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            ]
        elif self.model_id == "RealESR-general-x4v3":  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            netscale = 4
            file_url = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            ]
        else:
            raise ValueError(f"Model {self.model_id} not supported")

        root_dir = "data/upscaler"
        model_path = os.path.join("data/upscaler", self.model_id + ".pth")
        if not os.path.isfile(model_path):
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url,
                    model_dir=root_dir,
                    progress=True,
                    file_name=None,
                )

        # use dni to control the denoise strength
        dni_weight = None
        if self.model_id == "RealESR-general-x4v3" and self.denoise_strength != 1:
            wdn_model_path = model_path.replace(
                "realesr-general-x4v3", "realesr-general-wdn-x4v3"
            )
            model_path = [model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=config.api.data_type != "float32",
            gpu_id=self.gpu_id,
        )

    def unload(self):
        del self.upsampler
        self.upsampler = None
        self.memory_cleanup()

    def generate(self, job: Job) -> Image.Image:
        assert isinstance(job, UpscaleQueueEntry), "Wrong job type"
        input_image = convert_to_image(job.data.image)
        img = np.array(input_image)

        assert self.upsampler is not None, "Upsampler not loaded"
        output, _ = self.upsampler.enhance(img, outscale=job.data.upscale_factor)

        self.memory_cleanup()

        return Image.fromarray(output)
