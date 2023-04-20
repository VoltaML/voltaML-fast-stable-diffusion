from typing import Optional

import torch
from PIL import Image
from flamingo_mini import FlamingoModel, FlamingoProcessor

from core.utils import convert_to_image
from core.interrogation.clip import is_cpu
from core.interrogation.base_interrogator import InterrogationModel, InterrogationResult
from core.types import Job, InterrogatorQueueEntry


class FlamingoInterrogator(InterrogationModel):
    def __init__(self, device: str = "cuda", use_fp32: bool = False):
        super().__init__(device)

        self.device = device
        self.dtype = torch.float32 if use_fp32 else (torch.bfloat16 if is_cpu(device) else torch.float16)
        self.model: FlamingoModel
        self.processor = FlamingoProcessor
    
    def load(self):
        self.model = FlamingoModel.from_pretrained("")
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.processor = FlamingoProcessor(self.model.config)

    def unload(self):
        del self.model, self.processor
        self.memory_cleanup()

    def _infer(self, image: Image.Image, caption = Optional[str] = None):
        caption = caption if caption else "<image>"
        caption = self.model.generate_captions(
            self.processor,
            images=image,
            prompt=caption,
        )
        if isinstance(caption, list):
            caption = caption[0]
        self.memory_cleanup()
        return caption
    
    def generate(self, job: Job) -> InterrogationResult:
        if not isinstance(job, InterrogatorQueueEntry):
            return None  # type: ignore
        return InterrogationResult(self._infer(convert_to_image(job.data.image), job.data.caption), "") # type: ignore