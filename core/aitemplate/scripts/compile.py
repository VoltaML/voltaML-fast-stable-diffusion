#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import os
import shutil

import torch
from aitemplate.testing import detect_target
from diffusers import StableDiffusionPipeline

from api import websocket_manager
from api.websockets import Data, Notification
from core.config import config
from core.files import get_full_model_path

from ..src.compile_lib.compile_clip import compile_clip
from ..src.compile_lib.compile_unet import compile_unet
from ..src.compile_lib.compile_vae import compile_vae

logger = logging.getLogger(__name__)


def compile_diffusers(
    local_dir: str,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
):
    "Compile Stable Diffusion Pipeline to AITemplate format"

    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = StableDiffusionPipeline.from_pretrained(
        get_full_model_path(local_dir),
        revision="fp16",
        torch_dtype=torch.float16,
    )
    assert isinstance(pipe, StableDiffusionPipeline)
    pipe.to("cuda")

    assert (
        height % 64 == 0 and width % 64 == 0
    ), f"Height and Width must be multiples of 64, otherwise, the compilation process will fail. Got {height=} {width=}"

    ww = width // 8
    hh = height // 8

    dump_dir = os.path.join(
        "data",
        "aitemplate",
        local_dir.replace("/", "--") + f"__{width}x{height}x{batch_size}",
    )

    websocket_manager.broadcast_sync(
        Notification(
            severity="info",
            title="AITemplate",
            message=f"Compiling {local_dir} to AITemplate format",
        )
    )

    os.environ["NUM_BUILDERS"] = str(config.aitemplate.num_threads)

    # UNet
    websocket_manager.broadcast_sync(
        Data(
            data_type="aitemplate_compile",
            data={"unet": "process", "clip": "wait", "vae": "wait", "cleanup": "wait"},
        )
    )
    try:
        compile_unet(
            pipe.unet,  # type: ignore
            batch_size=batch_size * 2,
            width=ww,
            height=hh,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            hidden_dim=pipe.unet.config.cross_attention_dim,  # type: ignore
            attention_head_dim=pipe.unet.config.attention_head_dim,  # type: ignore
            dump_dir=dump_dir,
        )

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"unet": "finish"})
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error(e)
        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"unet": "error"})
        )
        websocket_manager.broadcast_sync(
            Notification(
                severity="error",
                title="AITemplate",
                message=f"Error while compiling UNet: {e}",
            )
        )
        raise e

    # CLIP
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"clip": "process"})
    )
    try:
        compile_clip(
            pipe.text_encoder,  # type: ignore
            batch_size=batch_size,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            depth=pipe.text_encoder.config.num_hidden_layers,  # type: ignore
            num_heads=pipe.text_encoder.config.num_attention_heads,  # type: ignore
            dim=pipe.text_encoder.config.hidden_size,  # type: ignore
            act_layer=pipe.text_encoder.config.hidden_act,  # type: ignore
            dump_dir=dump_dir,
        )

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"clip": "finish"})
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error(e)
        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"clip": "error"})
        )
        websocket_manager.broadcast_sync(
            Notification(
                severity="error",
                title="AITemplate",
                message=f"Error while compiling CLIP: {e}",
            )
        )
        raise e

    # VAE
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"vae": "process"})
    )
    try:
        compile_vae(
            pipe.vae,  # type: ignore
            batch_size=batch_size,
            width=ww,
            height=hh,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            dump_dir=dump_dir,
        )

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"vae": "finish"})
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error(e)
        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"vae": "error"})
        )
        websocket_manager.broadcast_sync(
            Notification(
                severity="error",
                title="AITemplate",
                message=f"Error while compiling VAE: {e}",
            )
        )
        raise e

    # Cleanup
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"cleanup": "process"})
    )

    try:
        # Clean all files except test.so recursively
        for root, _dirs, files in os.walk(dump_dir):
            for file in files:
                if file != "test.so":
                    os.remove(os.path.join(root, file))

        # Clean profiler (sometimes not present)
        try:
            shutil.rmtree(os.path.join(dump_dir, "profiler"))
        except FileNotFoundError:
            pass

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"cleanup": "finish"})
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error(e)
        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"cleanup": "error"})
        )
        websocket_manager.broadcast_sync(
            Notification(
                severity="error",
                title="AITemplate",
                message=f"Error while cleaning up: {e}",
            )
        )
        raise e

    websocket_manager.broadcast_sync(
        Notification(
            severity="success",
            title="AITemplate",
            message=f"Successfully compiled {local_dir} to AITemplate format",
        )
    )
