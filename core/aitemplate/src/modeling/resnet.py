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

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor, nn

from ..common import get_shape


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2dBias(
                channels, self.out_channels, 4, 2, 1, dtype=dtype
            )
        elif use_conv:
            conv = nn.Conv2dBias(self.channels, self.out_channels, 3, 1, 1, dtype=dtype)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, x: Tensor) -> Tensor:
        if self.use_conv_transpose:
            return self.conv(x)  # type: ignore

        x = nn.Upsampling2d(scale_factor=2.0, mode="nearest")(x)

        if self.use_conv:
            if self.name == "conv":
                x = self.conv(x)  # type: ignore
            else:
                x = self.Conv2d_0(x)  # type: ignore

        return x


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.dtype = dtype

        if use_conv:
            conv = nn.Conv2dBias(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                dtype=dtype,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_conv and self.padding == 0:
            shape = get_shape(hidden_states)
            padding = ops.full()([0, 1, 0, 0], 0.0, dtype=self.dtype)  # type: ignore
            padding._attrs["shape"][0] = shape[0]
            padding._attrs["shape"][2] = shape[2]
            padding._attrs["shape"][3] = shape[3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=1)  # type: ignore
            shape = get_shape(hidden_states)
            padding = ops.full()([0, 0, 1, 0], 0.0, dtype=self.dtype)  # type: ignore
            padding._attrs["shape"][0] = shape[0]
            padding._attrs["shape"][1] = shape[1]
            padding._attrs["shape"][3] = shape[3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=2)  # type: ignore

        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: Optional[int] = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        kernel: Optional[int] = None,
        output_scale_factor: float = 1.0,
        use_nin_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            use_swish=True,
            dtype=dtype,
        )

        self.conv1 = nn.Conv2dBias(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels, dtype=dtype)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            use_swish=True,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(dropout, dtype=dtype)  # type: ignore
        self.conv2 = nn.Conv2dBias(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        self.upsample = self.downsample = None

        self.use_nin_shortcut = (
            self.in_channels != self.out_channels
            if use_nin_shortcut is None
            else use_nin_shortcut
        )

        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2dBias(
                in_channels, out_channels, 1, 1, 0, dtype=dtype
            )
        else:
            self.conv_shortcut = None

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = x
        hidden_states = self.norm1(hidden_states)

        if self.upsample is not None:
            x = self.upsample(x)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        bs, _, _, dim = hidden_states.shape()
        if temb is not None:
            temb = self.time_emb_proj(ops.silu(temb))  # type: ignore
            bs, dim = temb.shape()  # type: ignore
            temb = ops.reshape()(temb, [bs, 1, 1, dim])  # type: ignore
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = hidden_states + x

        return out
