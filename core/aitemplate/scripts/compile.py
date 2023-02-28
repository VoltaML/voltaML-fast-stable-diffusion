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

import torch
from aitemplate.testing import detect_target
from diffusers import StableDiffusionPipeline

from ..src.compile_lib.compile_clip import compile_clip
from ..src.compile_lib.compile_unet import compile_unet
from ..src.compile_lib.compile_vae import compile_vae


def compile_diffusers(
    local_dir: str,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
):
    "Compile Stable Diffusion Pipeline to AITemplate format"

    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = StableDiffusionPipeline.from_pretrained(
        local_dir,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    assert isinstance(pipe, StableDiffusionPipeline)
    pipe.to("cuda")

    ww = width // 8
    hh = height // 8

    dump_dir = os.path.join("data", "aitemplate", local_dir)

    # CLIP
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
    # UNet
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
    # VAE
    compile_vae(
        pipe.vae,  # type: ignore
        batch_size=batch_size,
        width=ww,
        height=hh,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        dump_dir=dump_dir,
    )
