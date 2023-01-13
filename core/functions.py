import torch
from PIL.Image import Image

from api import websocket_manager
from api.websockets.data import Data
from core import shared
from core.inference.pytorch import PyTorchInferenceModel
from core.utils import convert_image_to_base64


def pytorch_callback(step: int, _: int, image_tensor: torch.FloatTensor):
    "Send a websocket message to the client with the progress percentage and partial image"

    current_model = shared.current_model
    assert isinstance(current_model, PyTorchInferenceModel)

    model = current_model.model
    assert model is not None

    # 8. Post-processing
    images = model.decode_latents(image_tensor)
    images: list[Image] = model.numpy_to_pil(images)
    image = images[0]

    assert isinstance(image, Image)

    websocket_manager.broadcast_sync(
        data=Data(
            data_type="txt2img",
            data={
                "progress": int((step / shared.current_steps) * 100),
                "image": convert_image_to_base64(image),
            },
        )
    )
