import os
from pathlib import Path
from typing import Any, Dict, List

from diffusers.utils.constants import hf_cache_home


class CachedModelList:
    "List of models downloaded for PyTorch and (or) converted to TRT"

    def __init__(self):
        self.pytorch_path = Path(hf_cache_home) / "diffusers"
        self.converted_path = Path("converted")

    def pytorch(self) -> list[Dict[str, Any]]:
        "List of models downloaded for PyTorch"

        models: List[Dict[str, Any]] = []
        for model_name in os.listdir(self.pytorch_path):

            # Skip if it is not a huggingface model
            if not "model" in model_name:
                continue

            name: str = "/".join(model_name.split("--")[1:3])
            models.append({"name": name, "path": name})

        for model_name in os.listdir(self.converted_path):
            models.append(
                {
                    "name": model_name,
                    "path": str(self.converted_path.joinpath(model_name)),
                }
            )

        return models
