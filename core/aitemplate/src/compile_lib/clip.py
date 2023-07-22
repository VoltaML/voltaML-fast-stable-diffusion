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

from ..modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
from ..common import mark_output

from ..modeling.mapping import map_clip


def compile_clip(
    pt_mod,
    batch_size=(1, 8),
    seqlen=64,
    dim=768,
    num_heads=12,
    depth=12,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    act_layer="gelu",
    constants=True,
    model_name="CLIPTextModel",
    work_dir="./tmp",
):
    mask_seq = 0
    causal = True

    ait_mod = ait_CLIPTextTransformer(
        num_hidden_layers=depth,
        hidden_size=dim,
        num_attention_heads=num_heads,
        batch_size=batch_size,
        seq_len=seqlen,
        causal=causal,
        mask_seq=mask_seq,
        act_layer=act_layer,
    )
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_clip(pt_mod)

    static_shape = batch_size[0] == batch_size[1]
    if static_shape:
        batch_size = batch_size[0]
    else:
        batch_size = IntVar(values=list(batch_size), name="batch_size")

    input_ids_ait = Tensor(
        [batch_size, seqlen], name="input0", dtype="int64", is_input=True  # type: ignore
    )
    position_ids_ait = Tensor(
        [batch_size, seqlen], name="input1", dtype="int64", is_input=True  # type: ignore
    )
    Y = ait_mod(input_ids=input_ids_ait, position_ids=position_ids_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, work_dir, model_name, constants=params_ait if constants else None
    )
