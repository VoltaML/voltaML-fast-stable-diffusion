from typing import Optional

import torch
from flamingo_mini import FlamingoModel, FlamingoProcessor
from PIL import Image

from core.interrogation.base_interrogator import InterrogationModel, InterrogationResult
from core.interrogation.clip import is_cpu
from core.types import InterrogatorQueueEntry, Job
from core.utils import convert_to_image


class FlamingoInterrogator(InterrogationModel):
    "Model that uses Flamingo Mini to generate image captions."

    def __init__(self, device: str = "cuda", use_fp32: bool = False):
        super().__init__(device)

        self.device = device
        self.dtype = (
            torch.float32
            if use_fp32
            else (torch.bfloat16 if is_cpu(device) else torch.float16)
        )
        self.model: FlamingoModel
        self.processor: FlamingoProcessor

    def load(self):
        model = FlamingoModel.from_pretrained("")
        assert isinstance(model, FlamingoModel)
        self.model = model
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.processor = FlamingoProcessor(self.model.config)

    def unload(self):
        del self.model, self.processor
        self.memory_cleanup()

    def _infer(self, image: Image.Image, caption: Optional[str] = None):
        caption = caption if caption else "<image>"
        output_caption = self.model.generate_captions(
            processor=self.processor,
            images=image,
            prompt=caption,
        )
        if isinstance(output_caption, list):
            output_caption = output_caption[0]

        # We are doing singe image caption only, so we can assert that the output is a string
        assert isinstance(output_caption, str)
        self.memory_cleanup()
        return output_caption

    def generate(self, job: Job) -> InterrogationResult:
        if not isinstance(job, InterrogatorQueueEntry):
            raise ValueError(
                "FlamingoInterrogator only supports InterrogatorQueueEntry"
            )
        return InterrogationResult(
            positive=[
                (
                    self._infer(convert_to_image(job.data.image), job.data.caption),
                    1,
                )
            ],
            negative=[],
        )
