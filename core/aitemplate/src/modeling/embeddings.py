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
import math

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor, nn


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    dtype: str = "float16",
    arange_name: str = "arange",
) -> Tensor:
    assert timesteps._rank() == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    exponent = (-math.log(max_period)) * Tensor(
        shape=[half_dim],
        dtype=dtype,
        name=arange_name,  # type: ignore
    )

    exponent = exponent * (1.0 / (half_dim - downscale_freq_shift))

    emb = ops.exp(exponent)
    emb = ops.reshape()(timesteps, [-1, 1]) * ops.reshape()(emb, [1, -1])

    emb = scale * emb

    if flip_sin_to_cos:
        emb = ops.concatenate()(
            [ops.cos(emb), ops.sin(emb)],
            dim=-1,
        )
    else:
        emb = ops.concatenate()(
            [ops.sin(emb), ops.cos(emb)],
            dim=-1,
        )
    return emb  # type: ignore


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        channel: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        dtype: str = "float16",
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(
            channel, time_embed_dim, specialization="swish", dtype=dtype
        )
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        dtype: str = "float16",
        arange_name: str = "arange",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.dtype = dtype
        self.arange_name = arange_name

    def forward(self, timesteps: Tensor) -> Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            dtype=self.dtype,
            arange_name=self.arange_name,
        )
        return t_emb
