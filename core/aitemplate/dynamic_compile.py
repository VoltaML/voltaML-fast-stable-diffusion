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
from typing import Optional, Tuple

import torch
from aitemplate.testing import detect_target
from diffusers import StableDiffusionPipeline
from rich.console import Console

from api import websocket_manager
from api.websockets.data import Data
from api.websockets.notification import Notification
from core.config import config
from core.inference.functions import load_pytorch_pipeline

from .src.compile_lib_dynamic.compile_clip_alt import compile_clip
from .src.compile_lib_dynamic.compile_unet_alt import compile_unet
from .src.compile_lib_dynamic.compile_vae_alt import compile_vae

console = Console()
logger = logging.getLogger(__name__)


def compile_diffusers(
    local_dir_or_id: str,
    width: Tuple[int, int] = (64, 2048),
    height: Tuple[int, int] = (64, 2048),
    batch_size: Tuple[int, int] = (1, 4),
    clip_chunks: int = 6,
    include_constants: Optional[bool] = None,
    convert_conv_to_gemm=True,
    device: str = "cuda",
):
    # Wipe out cache
    if os.path.exists("~/.aitemplate/cuda.db"):
        logger.info("Wiping out cache...")
        os.remove("~/.aitemplate/cuda.db")
        logger.info("Cache wiped out")

    torch.manual_seed(4896)
    start_time = time.time()

    use_fp16_acc = config.api.data_type != "float32"

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    assert (
        width[0] % 64 == 0 and width[1] % 64 == 0
    ), "Minimum Width and Maximum Width must be multiples of 64, otherwise, the compilation process will fail."
    assert (
        height[0] % 64 == 0 and height[1] % 64 == 0
    ), "Minimum Height and Maximum Height must be multiples of 64, otherwise, the compilation process will fail."

    pipe = load_pytorch_pipeline(
        model_id_or_path=local_dir_or_id, device=device, optimize=False
    )
    assert isinstance(pipe, StableDiffusionPipeline)
    pipe.to("cuda")

    dump_dir = os.path.join(
        "data",
        "aitemplate",
        local_dir_or_id.replace("/", "--") + "__dynamic",
    )

    os.environ["NUM_BUILDERS"] = str(config.aitemplate.num_threads)

    websocket_manager.broadcast_sync(
        Notification(
            severity="info",
            title="AITemplate",
            message=f"Compiling {local_dir_or_id} to AITemplate format",
        )
    )

    # CLIP
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"clip": "process"})
    )
    with console.status("[bold green]Compiling CLIP..."):
        compile_clip(
            pipe.text_encoder,
            dump_dir=dump_dir,
            batch_size=batch_size,
            seqlen=77,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            depth=pipe.text_encoder.config.num_hidden_layers,
            num_heads=pipe.text_encoder.config.num_attention_heads,
            dim=pipe.text_encoder.config.hidden_size,
            act_layer=pipe.text_encoder.config.hidden_act,
            constants=True if include_constants else False,
        )

        websocket_manager.broadcast_sync(
            Data(data_type="aitemplate_compile", data={"clip": "finish"})
        )
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
    with console.status("[bold green]Compiling UNet..."):
        compile_unet(
            pipe.unet,
            dump_dir=dump_dir,
            batch_size=batch_size,
            width=width,
            height=height,
            clip_chunks=clip_chunks,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            hidden_dim=pipe.unet.config.cross_attention_dim,
            attention_head_dim=pipe.unet.config.attention_head_dim,
            use_linear_projection=pipe.unet.config.get("use_linear_projection", False),
            constants=True if include_constants else False,
        )

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

    # VAE
    with console.status("[bold green]Compiling VAE..."):
        compile_vae(
            pipe.vae,
            dump_dir=dump_dir,
            batch_size=batch_size,
            width=width,
            height=height,
            use_fp16_acc=use_fp16_acc,
            convert_conv_to_gemm=convert_conv_to_gemm,
            constants=True if include_constants else False,
        )

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

    # Cleanup
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"cleanup": "process"})
    )

    with console.status("[bold green]Cleaning up..."):
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

    with console.status("[bold green]Releasing memory"):
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
