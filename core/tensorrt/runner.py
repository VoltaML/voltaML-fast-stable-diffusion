# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from PIL.Image import Image


class BaseRunner(ABC):
    "Abstract class for the TensorRT model"

    loading = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def activate(self) -> None:
        "Load the model into memory"

    @abstractmethod
    def teardown(self) -> None:
        "Unload the model from memory"

    def wait_loading(self):
        "Wait for the model to finish"

        if not self.loading:
            return
        while self.loading:
            time.sleep(0.1)

    @abstractmethod
    def infer(
        self,
        prompt: str,
        negative_prompt: str,
        batch_size: int,
        batch_count: int,
        scheduler_id: str,
        steps: int,
        scale: int,
        image_height: int,
        image_width: int,
        seed: int,
    ) -> Tuple[List[Tuple[List[Image], Dict, Dict]], float]:
        "Generate images from the model"
