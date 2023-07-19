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
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.controlnet import (
    ControlNetModel as ait_ControlNetModel,
)
from ..common import mark_output

from ..modeling.mapping import map_controlnet


def compile_controlnet(
    pt_mod,
    batch_size=(1, 4),
    height=(64, 2048),
    width=(64, 2048),
    clip_chunks=1,
    dim=320,
    hidden_dim=768,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    model_name="ControlNetModel",
    constants=False,
    work_dir="./tmp",
    down_factor=8,
):
    ait_mod = ait_ControlNetModel()
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_controlnet(pt_mod, dim=dim)

    static_shape = (
        width[0] == width[1]
        and height[0] == height[1]
        and batch_size[0] == batch_size[1]
    )

    if static_shape:
        batch_size = batch_size[0] * 2  # double batch size for unet
        height_d = height[0] // down_factor
        width_d = width[0] // down_factor
        height_c = height[0]
        width_c = width[0]
        clip_chunks = 77
        embedding_size = clip_chunks
    else:
        batch_size = batch_size[0], batch_size[1] * 2  # double batch size for unet
        batch_size = IntVar(values=list(batch_size), name="batch_size")
        height_d = height[0] // down_factor, height[1] // down_factor
        height_d = IntVar(values=list(height_d), name="height_d")
        width_d = width[0] // down_factor, width[1] // down_factor
        width_d = IntVar(values=list(width_d), name="width_d")
        height_c = height
        height_c = IntVar(values=list(height_c), name="height_c")
        width_c = width
        width_c = IntVar(values=list(width_c), name="width_c")
        clip_chunks = 77, 77 * clip_chunks
        embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, 4], name="input0", is_input=True  # type: ignore
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim], name="input2", is_input=True  # type: ignore
    )
    controlnet_condition_ait = Tensor(
        [batch_size, height_c, width_c, 3], name="input3", is_input=True  # type: ignore
    )

    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        text_embeddings_pt_ait,
        controlnet_condition_ait,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, work_dir, model_name, constants=params_ait if constants else None
    )
