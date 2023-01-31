import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from core.config import config

logger = logging.getLogger(__name__)


class CachedModelList:
    "List of models downloaded for PyTorch and (or) converted to TRT"

    def __init__(self):
        self.pytorch_path = Path(config.cache_dir)
        self.checkpoint_converted_path = Path("converted")
        self.tensorrt_engine_path = Path(
            os.environ.get("TENSORRT_ENGINE_PATH", "engine")
        )

    def pytorch(self) -> List[Dict[str, Any]]:
        "List of models downloaded for PyTorch"

        models: List[Dict[str, Any]] = []

        logger.debug(f"Looking for PyTorch models in {self.pytorch_path}")
        for model_name in os.listdir(self.pytorch_path):
            logger.debug(f"Found model {model_name}")

            # Skip if it is not a huggingface model
            if not "model" in model_name:
                continue

            name: str = "/".join(model_name.split("--")[1:3])
            models.append({"name": name, "path": name, "backend": "PyTorch"})

        logger.debug(
            f"Looking for converted models in {self.checkpoint_converted_path}"
        )
        for model_name in os.listdir(self.checkpoint_converted_path):
            logger.debug(f"Found model {model_name}")

            models.append(
                {
                    "name": model_name,
                    "path": str(self.checkpoint_converted_path.joinpath(model_name)),
                    "backend": "PyTorch",
                }
            )

        return models

    def tensorrt(self) -> List[Dict[str, Any]]:
        "List of models converted to TRT"

        models: List[Dict[str, Any]] = []

        logger.debug(f"Looking for TensorRT models in {self.tensorrt_engine_path}")

        for author in os.listdir(self.tensorrt_engine_path):
            logger.debug(f"Found author {author}")
            for model_name in os.listdir(self.tensorrt_engine_path.joinpath(author)):
                logger.debug(f"Found model {model_name}")
                models.append(
                    {
                        "name": "/".join([author, model_name]),
                        "path": "/".join([author, model_name]),
                        "backend": "TensorRT",
                    }
                )

        return models

    def all(self):
        "List both PyTorch and TensorRT models"

        return self.pytorch() + self.tensorrt()
