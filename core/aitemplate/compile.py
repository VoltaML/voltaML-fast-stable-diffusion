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
from typing import Tuple, Union

import torch
from aitemplate.testing import detect_target

from api import websocket_manager
from api.websockets import Data, Notification
from core.config import config
from core.inference.functions import load_pytorch_pipeline

from .src.compile_lib.clip import compile_clip
from .src.compile_lib.unet import compile_unet
from .src.compile_lib.vae import compile_vae

logger = logging.getLogger(__name__)


def compile_diffusers(
    local_dir_or_id: str,
    width: Union[int, Tuple[int, int]] = 512,
    height: Union[int, Tuple[int, int]] = 512,
    batch_size: Union[int, Tuple[int, int]] = 1,
    clip_chunks: int = 6,
    convert_conv_to_gemm=True,
    invalidate_cache=False,
    device: str = "cuda",
):
    "Compile Stable Diffusion Pipeline to AITemplate format"

    # Wipe out cache
    if os.path.exists("~/.aitemplate/cuda.db"):
        logger.info("Wiping out cache...")
        os.remove("~/.aitemplate/cuda.db")
        logger.info("Cache wiped out")

    use_fp16_acc = config.api.data_type != "float32"
    start_time = time.time()

    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = load_pytorch_pipeline(
        model_id_or_path=local_dir_or_id,
        device=device,
    )

    if isinstance(width, int):
        width = (width, width)
    if isinstance(height, int):
        height = (height, height)
    if isinstance(batch_size, int):
        batch_size = (batch_size, batch_size)

    assert (
        height[0] % 64 == 0
        and height[1] % 64 == 0
        and width[0] % 64 == 0
        and width[1] % 64 == 0
    ), f"Height and Width must be multiples of 64, otherwise, the compilation process will fail. Got {height=} {width=}"

    dump_dir = os.path.join(
        "data",
        "aitemplate",
        local_dir_or_id.replace("/", "--")
        + f"__{width[0]}-{width[1]}x{height[0]}-{height[1]}x{batch_size[0]}-{batch_size[1]}",
    )

    websocket_manager.broadcast_sync(
        Notification(
            severity="info",
            title="AITemplate",
            message=f"Compiling {local_dir_or_id} to AITemplate format",
        )
    )

    os.environ["NUM_BUILDERS"] = str(config.aitemplate.num_threads)

    websocket_manager.broadcast_sync(
        Data(
            data_type="aitemplate_compile",
            data={
                "clip": "wait",
                "unet": "wait",
                "controlnet_unet": "wait",
                "vae": "wait",
                "cleanup": "wait",
            },
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
                seqlen=pipe.text_encoder.config.max_position_embeddings,
                use_fp16_acc=use_fp16_acc,
                constants=True,
                convert_conv_to_gemm=convert_conv_to_gemm,
                depth=pipe.text_encoder.config.num_hidden_layers,  # type: ignore
                num_heads=pipe.text_encoder.config.num_attention_heads,  # type: ignore
                dim=pipe.text_encoder.config.hidden_size,  # type: ignore
                act_layer=pipe.text_encoder.config.hidden_act,  # type: ignore
                work_dir=dump_dir,
            )

            websocket_manager.broadcast_sync(
                Data(data_type="aitemplate_compile", data={"clip": "finish"})
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

    # UNet
    websocket_manager.broadcast_sync(
        Data(data_type="aitemplate_compile", data={"unet": "process"})
    )
    try:
        if (
            invalidate_cache
            or not Path(dump_dir).joinpath("UNet2DConditionModel/test.so").exists()
        ):
            compile_unet(
                pipe.unet,  # type: ignore
                batch_size=batch_size,
                width=width,
                height=height,
                use_fp16_acc=use_fp16_acc,
                work_dir=dump_dir,
                convert_conv_to_gemm=convert_conv_to_gemm,
                hidden_dim=pipe.unet.config.cross_attention_dim,
                attention_head_dim=pipe.unet.config.attention_head_dim,
                use_linear_projection=pipe.unet.config.get(
                    "use_linear_projection", False
                ),
                block_out_channels=pipe.unet.config.block_out_channels,
                down_block_types=pipe.unet.config.down_block_types,
                up_block_types=pipe.unet.config.up_block_types,
                in_channels=pipe.unet.config.in_channels,
                out_channels=pipe.unet.config.out_channels,
                class_embed_type=pipe.unet.config.class_embed_type,
                num_class_embeds=pipe.unet.config.num_class_embeds,
                only_cross_attention=pipe.unet.config.only_cross_attention,
                sample_size=pipe.unet.config.sample_size,
                dim=pipe.unet.config.block_out_channels[0],
                time_embedding_dim=None,
                down_factor=8,
                clip_chunks=clip_chunks,
                constants=True,
                controlnet=False,
                conv_in_kernel=pipe.unet.config.conv_in_kernel,
                projection_class_embeddings_input_dim=pipe.unet.config.projection_class_embeddings_input_dim,
                addition_embed_type=pipe.unet.config.addition_embed_type,
                addition_time_embed_dim=pipe.unet.config.addition_time_embed_dim,
                transformer_layers_per_block=pipe.unet.config.transformer_layers_per_block,
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
            compile_unet(
                pipe.unet,  # type: ignore
                model_name="ControlNetUNet2DConditionModel",
                batch_size=batch_size,
                width=width,
                height=height,
                use_fp16_acc=use_fp16_acc,
                work_dir=dump_dir,
                convert_conv_to_gemm=convert_conv_to_gemm,
                hidden_dim=pipe.unet.config.cross_attention_dim,
                attention_head_dim=pipe.unet.config.attention_head_dim,
                use_linear_projection=pipe.unet.config.get(
                    "use_linear_projection", False
                ),
                block_out_channels=pipe.unet.config.block_out_channels,
                down_block_types=pipe.unet.config.down_block_types,
                up_block_types=pipe.unet.config.up_block_types,
                in_channels=pipe.unet.config.in_channels,
                out_channels=pipe.unet.config.out_channels,
                class_embed_type=pipe.unet.config.class_embed_type,
                num_class_embeds=pipe.unet.config.num_class_embeds,
                only_cross_attention=pipe.unet.config.only_cross_attention,
                sample_size=pipe.unet.config.sample_size,
                dim=pipe.unet.config.block_out_channels[0],
                time_embedding_dim=None,
                down_factor=8,
                constants=True,
                controlnet=True,
                conv_in_kernel=pipe.unet.config.conv_in_kernel,
                projection_class_embeddings_input_dim=pipe.unet.config.projection_class_embeddings_input_dim,
                addition_embed_type=pipe.unet.config.addition_embed_type,
                addition_time_embed_dim=pipe.unet.config.addition_time_embed_dim,
                transformer_layers_per_block=pipe.unet.config.transformer_layers_per_block,
            )
        else:
            logger.info("UNet already compiled. Skipping...")

        # Dump UNet config
        with open(
            os.path.join(dump_dir, "ControlNetUNet2DConditionModel", "config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(pipe.unet.config, f, indent=4, ensure_ascii=False)  # type: ignore
            logger.info("UNet config saved")

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
                width=width,
                height=height,
                use_fp16_acc=use_fp16_acc,
                convert_conv_to_gemm=convert_conv_to_gemm,
                block_out_channels=pipe.vae.config.block_out_channels,
                layers_per_block=pipe.vae.config.layers_per_block,
                act_fn=pipe.vae.config.act_fn,
                latent_channels=pipe.vae.config.latent_channels,
                in_channels=pipe.vae.config.in_channels,
                out_channels=pipe.vae.config.out_channels,
                down_block_types=pipe.vae.config.down_block_types,
                up_block_types=pipe.vae.config.up_block_types,
                sample_size=pipe.vae.config.sample_size,
                input_size=(64, 64),
                down_factor=8,
                vae_encode=False,
                constants=True,
                work_dir=dump_dir,
                dtype="float16" if use_fp16_acc else "float32",
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
            timeout=0,
        )
    )

    logger.info(f"Finished compiling in {deltatime:.2f} seconds")
