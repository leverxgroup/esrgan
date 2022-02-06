import collections
from typing import Callable, List, Tuple

import torch
from torch import nn

from esrgan import utils
from esrgan.nn import modules

__all__ = ["ESREncoder", "ESRNetDecoder"]


class ESREncoder(nn.Module):
    """'Encoder' part of ESRGAN network, processing images in LR space.

    It has been proposed in `ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks`_.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the encoder.
        growth_channels: Number of channels in the latent space.
        num_basic_blocks: Depth of the encoder, number of Residual-in-Residual
            Dense block (RRDB) to use.
        num_dense_blocks: Number of dense blocks to use to form `RRDB` block.
        num_residual_blocks: Number of convolutions to use to form dense block.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use e.g., :py:class:`nn.ReLU`.
        residual_scaling: Residual connections scaling factor.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        growth_channels: int = 32,
        num_basic_blocks: int = 23,
        num_dense_blocks: int = 3,
        num_residual_blocks: int = 5,
        conv: Callable[..., nn.Module] = modules.Conv2d,
        activation: Callable[..., nn.Module] = modules.LeakyReLU,
        residual_scaling: float = 0.2,
    ) -> None:
        super().__init__()

        blocks_list: List[nn.Module] = []

        # first conv
        first_conv = conv(in_channels, out_channels)
        blocks_list.append(first_conv)

        # basic blocks - sequence of rrdb layers
        for _ in range(num_basic_blocks):
            basic_block = modules.ResidualInResidualDenseBlock(
                num_features=out_channels,
                growth_channels=growth_channels,
                conv=conv,
                activation=activation,
                num_dense_blocks=num_dense_blocks,
                num_blocks=num_residual_blocks,
                residual_scaling=residual_scaling,
            )
            blocks_list.append(basic_block)

        # last conv of the encoder
        last_conv = conv(out_channels, out_channels)
        blocks_list.append(last_conv)

        self.blocks = nn.ModuleList(blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of images.

        Returns:
            Batch of embeddings.

        """
        input_ = output = self.blocks[0](x)
        for module in self.blocks[1:]:
            output = module(output)

        return input_ + output


class ESRNetDecoder(nn.Module):
    """'Decoder' part of ESRGAN, converting embeddings to output image.

    It has been proposed in `ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks`_.

    Args:
        in_channels: Number of channels in the input embedding.
        out_channels: Number of channels in the output image.
        scale_factor: Ratio between the size of the high-resolution image
            (output) and its low-resolution counterpart (input).
            In other words multiplier for spatial size.
        conv: Class constructor or partial object which when called
            should return convolutional layer e.g., :py:class:`nn.Conv2d`.
        activation: Class constructor or partial object which when called
            should return activation function to use e.g., :py:class:`nn.ReLU`.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        scale_factor: int = 2,
        conv: Callable[..., nn.Module] = modules.Conv2d,
        activation: Callable[..., nn.Module] = modules.LeakyReLU,
    ) -> None:
        super().__init__()

        # check params
        if utils.is_power_of_two(scale_factor):
            raise NotImplementedError(
                f"scale_factor should be power of 2, got {scale_factor}"
            )

        blocks_list: List[Tuple[str, nn.Module]] = []

        # upsampling
        for i in range(scale_factor // 2):
            upsampling_block = modules.InterpolateConv(
                num_features=in_channels,
                conv=conv,
                activation=activation,
            )
            blocks_list.append((f"upsampling_{i}", upsampling_block))

        # highres conv + last conv
        last_conv = nn.Sequential(
            conv(in_channels, in_channels),
            activation(),
            conv(in_channels, out_channels),
        )
        blocks_list.append(("conv", last_conv))

        self.blocks = nn.Sequential(collections.OrderedDict(blocks_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of embeddings.

        Returns:
            Batch of upscaled images.

        """
        output = self.blocks(x)

        return output
