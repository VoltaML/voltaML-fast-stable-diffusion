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
import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch
from aitemplate.testing import detect_target

from api import websocket_manager
from api.websockets import Data, Notification
from core.config import config
from core.inference.functions import load_pytorch_pipeline

from .src.compile_lib.compile_clip import compile_clip
from .src.compile_lib.compile_controlnet_unet import compile_controlnet_unet
from .src.compile_lib.compile_unet import compile_unet
from .src.compile_lib.compile_vae import compile_vae

logger = logging.getLogger(__name__)


def compile_diffusers(
    local_dir_or_id: str,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    convert_conv_to_gemm=True,
    invalidate_cache=False,
    device: str = "cuda",
):
    "Compile Stable Diffusion Pipeline to AITemplate format"

    use_fp16_acc = not config.api.use_fp32
    start_time = time.time()

    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = load_pytorch_pipeline(
        model_id_or_path=local_dir_or_id,
        device=device,
    )

    assert (
        height % 64 == 0 and width % 64 == 0
    ), f"Height and Width must be multiples of 64, otherwise, the compilation process will fail. Got {height=} {width=}"

    ww = width // 8
    hh = height // 8

    dump_dir = os.path.join(
        "data",
        "aitemplate",
        local_dir_or_id.replace("/", "--") + f"__{width}x{height}x{batch_size}",
    )

    websocket_manager.broadcast_sync(
        Notification(
            severity="info",
            title="AITemplate",
            message=f"Compiling {local_dir_or_id} to AITemplate format",
        )
    )

    os.environ["NUM_BUILDERS"] = str(config.aitemplate.num_threads)

    # UNet
    websocket_manager.broadcast_sync(
        Data(
            data_type="aitemplate_compile",
            data={
                "unet": "process",
                "controlnet_unet": "wait",
                "clip": "wait",
                "vae": "wait",
                "cleanup": "wait",
            },
        )
    )
    try:
        if (
            invalidate_cache
            or not Path(dump_dir).joinpath("UNet2DConditionModel/test.so").exists()
        ):
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
        else:
            logger.info("UNet already compiled. Skipping...")

        # Dump UNet config
        with open(
            os.path.join(dump_dir, "UNet2DConditionModel", "config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(pipe.unet.config, f, indent=4, ensure_ascii=False)  # type: ignore
            logger.info("UNet config saved")

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

    # ControlNet UNet
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"controlnet_unet": "process"})
    )
    try:
        if (
            invalidate_cache
            or not Path(dump_dir)
            .joinpath("ControlNetUNet2DConditionModel/test.so")
            .exists()
        ):
            compile_controlnet_unet(
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
        else:
            logger.info("ControlNet UNet already compiled. Skipping...")

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"controlnet_unet": "finish"})
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error(e)
        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"controlnet_unet": "error"})
        )
        websocket_manager.broadcast_sync(
            Notification(
                severity="error",
                title="AITemplate",
                message=f"Error while compiling ControlNet UNet: {e}",
            )
        )

    # CLIP
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"clip": "process"})
    )
    try:
        if (
            invalidate_cache
            or not Path(dump_dir).joinpath("CLIPTextModel/test.so").exists()
        ):
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
        else:
            logger.info("CLIP already compiled. Skipping...")

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
        if (
            invalidate_cache
            or not Path(dump_dir).joinpath("AutoencoderKL/test.so").exists()
        ):
            compile_vae(
                pipe.vae,  # type: ignore
                batch_size=batch_size,
                width=ww,
                height=hh,
                use_fp16_acc=use_fp16_acc,
                convert_conv_to_gemm=convert_conv_to_gemm,
                dump_dir=dump_dir,
            )
        else:
            logger.info("VAE already compiled. Skipping...")

        # Dump VAE config
        with open(
            os.path.join(dump_dir, "AutoencoderKL", "config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(pipe.vae.config, f, indent=4, ensure_ascii=False)  # type: ignore
            logger.info("VAE config saved")

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
                if file not in ["test.so", "config.json"]:
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

    del pipe
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    deltatime = time.time() - start_time

    websocket_manager.broadcast_sync(
        Notification(
            severity="success",
            title="AITemplate",
            message=f"Successfully compiled {local_dir_or_id} to AITemplate format in {deltatime:.2f} seconds",
        )
    )

    logger.info(f"Finished compiling in {deltatime:.2f} seconds")
