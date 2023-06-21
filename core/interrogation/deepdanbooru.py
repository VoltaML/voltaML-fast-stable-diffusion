from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.backend_config.tensorrt import (
    get_tensorrt_backend_config_dict,
)
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    convert_to_reference_fx,
    prepare_fx,
)
from tqdm import tqdm

from core.config import config
from core.interrogation.base_interrogator import InterrogationModel, InterrogationResult
from core.interrogation.clip import is_cpu
from core.interrogation.models.deepdanbooru_model import DeepDanbooruModel
from core.types import InterrogatorQueueEntry, Job
from core.utils import convert_to_image, download_file

DEEPDANBOORU_URL = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt"


class DeepdanbooruInterrogator(InterrogationModel):
    "Interrogator that will generate captions for images (Anime)"

    def __init__(
        self,
        device: str = "cuda",
        quantized: bool = False,
        autoload: bool = False,
    ):
        super().__init__(device)

        self.tags = []
        self.model: DeepDanbooruModel
        self.model_location = Path("data") / "models" / "deepdanbooru.pt"
        self.dtype = torch.quint8 if quantized else config.api.dtype
        self.device: torch.device
        if isinstance(self.device, str):
            self.device = torch.device(device)
        else:
            self.device = device  # type: ignore
        self.quantized = quantized
        if autoload:
            self.load()

    def load(self):
        self.model = DeepDanbooruModel()
        if not self.model_location.exists():
            download_file(DEEPDANBOORU_URL, self.model_location)
        state_dict = torch.load(self.model_location, map_location="cpu")
        self.tags = state_dict.get("tags", [])
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Quantize if needed (should support TRT too)
        if self.quantized:
            qconfig_dict = get_default_qconfig_mapping()

            # Images will be ( forcefully :) ) resized to 512x512 with only 3 channels, so [1, 512, 512, 3]'s the shape
            prepared = prepare_fx(
                self.model,
                qconfig_dict,
                (torch.randn(1, 512, 512, 3),),  # great syntax pytorch :)
            )
            for _ in tqdm(range(25), unit="it", unit_scale=False, desc="Warming up"):
                prepared(torch.randn(1, 512, 512, 3))

            if is_cpu(self.device):
                self.model = convert_fx(prepared)  # type: ignore
            else:
                try:
                    self.model = convert_to_reference_fx(prepared, backend_config=get_tensorrt_backend_config_dict())  # type: ignore
                except Exception:  # pylint: disable=broad-except
                    pass
            del prepared
        else:
            self.model.to(self.device, dtype=self.dtype)  # type: ignore

    def _infer(
        self, image: Image.Image, sort: bool = False, threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        pic = image.convert("RGB").resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad():
            x = torch.from_numpy(a).to(
                device=self.device,
                dtype=self.dtype,
            )
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}
        for tag, probability in zip(self.tags, y):  # type: ignore
            if probability < threshold:
                continue
            if tag.startswith("rating:"):
                continue
            probability_dict[tag] = probability

        if sort:
            tags = sorted(probability_dict)
        else:
            tags = [
                tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])
            ]

        self.memory_cleanup()

        output = []
        for tag in tags:
            probability = probability_dict[tag]
            output.append((tag, float(probability)))

        return output

    def generate(self, job: Job) -> InterrogationResult:
        if not isinstance(job, InterrogatorQueueEntry):
            raise ValueError(
                "DeepdanbooruInterrogator only supports InterrogatorQueueEntry"
            )
        return InterrogationResult(
            self._infer(convert_to_image(job.data.image), threshold=job.data.threshold),
            [],
        )

    def unload(self):
        del self.model
        self.memory_cleanup()
