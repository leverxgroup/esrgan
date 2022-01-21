from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from esrgan.model.module.blocks.misc import Conv2d, LeakyReLU


class SubPixelConv(nn.Module):
    """Rearranges elements in a tensor of shape
    :math:`(B, C \\times r^2, H, W)` to a tensor of shape
    :math:`(B, C, H \\times r, W \\times r)`.

    Look at the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network`_
    for more details.

    Args:
        num_features: Number of channels in the input tensor.
        scale_factor: Factor to increase spatial resolution by.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use after
            sub-pixel convolution e.g., :py:class:`nn.PReLU`.

    .. _`Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network`:
        https://arxiv.org/pdf/1609.05158.pdf

    """

    def __init__(
        self,
        num_features: int,
        scale_factor: int = 2,
        conv: Callable[..., nn.Module] = Conv2d,
        activation: Callable[..., nn.Module] = nn.PReLU,
    ):
        super().__init__()

        self.block = nn.Sequential(
            conv(num_features, num_features * 4),
            nn.PixelShuffle(upscale_factor=scale_factor),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Apply conv -> shuffle pixels -> apply nonlinearity.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled input.

        """
        output = self.block(x)

        return output


class InterpolateConv(nn.Module):
    """Upsamples a given multi-channel 2D (spatial) data.

    Args:
        num_features: Number of channels in the input tensor.
        scale_factor: Factor to increase spatial resolution by.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use after convolution
            e.g., :py:class:`nn.PReLU`.

    """

    def __init__(
        self,
        num_features: int,
        scale_factor: int = 2,
        conv: Callable[..., nn.Module] = Conv2d,
        activation: Callable[..., nn.Module] = LeakyReLU,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.block = nn.Sequential(
            conv(num_features, num_features),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Upscale input -> apply conv -> apply nonlinearity.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled data.

        """
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        output = self.block(x)

        return output


__all__ = ["SubPixelConv", "InterpolateConv"]
