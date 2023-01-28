import os
from pathlib import Path
from typing import Dict

from diffusers.utils.constants import hf_cache_home


class CachedModelList:
    "List of models downloaded for PyTorch and (or) converted to TRT"

    def __init__(self):
        self.pytorch_path = Path(hf_cache_home) / "diffusers"

    def pytorch(self) -> Dict[str, Path]:
        "List of models downloaded for PyTorch"

        models: Dict[str, Path] = {}
        for model_name in os.listdir(self.pytorch_path):

            # Skip if it is not a huggingface model
            if not "model" in model_name:
                continue

            name: str = "/".join(model_name.split("--")[1:3])
            models[name] = self.pytorch_path.joinpath(
                model_name
            )  # pylint: disable=modified-iterating-dict
        return models
