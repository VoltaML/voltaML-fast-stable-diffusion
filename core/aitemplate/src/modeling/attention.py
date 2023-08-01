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
from typing import Optional

from aitemplate.compiler.ops import reshape
from aitemplate.frontend import Tensor, nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        dtype="float16",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_groups, channels, eps, dtype=dtype)
        self.attention = nn.CrossAttention(
            channels,
            height * width,
            height * width,
            self.num_heads,
            qkv_bias=True,
            dtype=dtype,
        )
        self.rescale_output_factor = rescale_output_factor

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states

        hidden_states = self.group_norm(hidden_states)
        o_shape = hidden_states.shape()
        batch_dim = o_shape[0]

        hidden_states = reshape()(
            hidden_states,
            [batch_dim, -1, self.channels],
        )

        res = self.attention(hidden_states, hidden_states, hidden_states, residual) * (
            1 / self.rescale_output_factor
        )

        res = reshape()(res, o_shape)
        return res
