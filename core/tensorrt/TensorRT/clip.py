# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import os

from .engine import build_engine
from .models import CLIP


def create_clip_engine():
    model_id = "openai/clip-vit-large-patch14"
    model_data = CLIP(
        model_id,
        device="cuda",
        verbose=False,
        max_batch_size=1,
    )
    model_dir = os.path.join("engine", model_id.replace("/", os.sep))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "engine"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "onnx"), exist_ok=True)
    engine = build_engine(
        "clip",
        os.path.join(model_dir, "engine"),
        os.path.join(model_dir, "onnx"),
        model_data,
    )
    return engine
