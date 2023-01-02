from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Txt2imgJob:
    prompt: str
    id: str = field(default="None")
    negative_prompt: str = field(default="")
    width: int = field(default=512)
    height: int = field(default=512)
    steps: int = field(default=25)
    guidance_scale: float = field(default=7)
    seed: int = field(default=0)
    batch_size: int = 1
    batch_count: int = 1
    model_path: str = ""
    backend: Literal["TensorRT", "PyTorch"] = "PyTorch"
