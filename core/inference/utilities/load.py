from typing import Dict
from io import BytesIO
import logging
from time import perf_counter as time

import torch
from safetensors.torch import load, load_file

from core.config import config


logger = logging.getLogger(__name__)


def load_checkpoint(path: str, from_safetensors: bool) -> Dict[str, torch.Tensor]:
    now = time()
    if from_safetensors:
        dev = str(config.api.load_device)
        if "cuda" in dev:
            dev = int(dev.split(":")[1])

        if config.api.stream_load:
            with open(path, "rb") as f:
                checkpoint = load(f.read())
            checkpoint = {
                k: v.to(device=config.api.load_device) for k, v in checkpoint.items()
            }
        else:
            checkpoint = load_file(path, device=dev)  # type: ignore

    else:
        if config.api.stream_load:
            with open(path, "rb") as f:
                buffer = BytesIO(f.read())
                checkpoint = torch.load(buffer, map_location=config.api.load_device)
        else:
            try:
                checkpoint = torch.load(
                    path,
                    mmap=True,
                    weights_only=True,
                    map_location=config.api.load_device,
                )
            except RuntimeError:
                # File is really old / wasn't saved with "_use_new_zipfile_serialization=True"
                # so we cannot mmap.
                checkpoint = torch.load(
                    path,
                    mmap=False,
                    weights_only=True,
                    map_location=config.api.load_device,
                )
    logger.debug(f'Loading "{path}" took {round(time() - now, 2)}s.')
    return checkpoint
