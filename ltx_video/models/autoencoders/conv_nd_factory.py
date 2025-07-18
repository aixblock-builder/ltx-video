from typing import Tuple, Union

import torch

from ltx_video.models.autoencoders.dual_conv3d import DualConv3d
from ltx_video.models.autoencoders.causal_conv3d import CausalConv3d


def make_conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    causal=False,
):
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias=True,
):
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    elif dims == 3 or dims == (2, 1):
        return torch.nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
