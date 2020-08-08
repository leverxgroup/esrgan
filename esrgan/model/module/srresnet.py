import collections
from typing import List, Tuple

import torch
from torch import nn

from esrgan import utils
from esrgan.model.module import blocks
from esrgan.utils.types import ModuleParams


class SRResNetEncoder(nn.Module):
    """'Encoder' part of SRResNet network, processing images in LR space.

    It has been proposed in `Photo-Realistic Single Image Super-Resolution
    Using a Generative Adversarial Network`_.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the encoder.
        num_basic_blocks: Depth of the encoder, number of basic blocks to use.
        conv_fn: Convolutional layers parameters.
        norm_fn: Batch norm layer to use.
        activation_fn: Activation function to use after BN layers.

    .. _`Photo-Realistic Single Image Super-Resolution Using a Generative
        Adversarial Network`: https://arxiv.org/pdf/1609.04802.pdf

    """

    @utils.process_fn_params
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_basic_blocks: int = 16,
        conv_fn: ModuleParams = blocks.Conv2d,
        norm_fn: ModuleParams = nn.BatchNorm2d,
        activation_fn: ModuleParams = nn.PReLU,
    ) -> None:
        super().__init__()

        num_features = out_channels
        blocks_list: List[nn.Module] = []

        # first conv
        first_conv = nn.Sequential(
            conv_fn(in_channels, num_features), activation_fn()
        )
        blocks_list.append(first_conv)

        # basic blocks - sequence of B residual blocks
        for _ in range(num_basic_blocks):
            basic_block = nn.Sequential(
                conv_fn(num_features, num_features),
                norm_fn(num_features,),
                activation_fn(),
                conv_fn(num_features, num_features),
                norm_fn(num_features),
            )
            blocks_list.append(blocks.ResidualModule(basic_block))

        # last conv of the encoder
        last_conv = nn.Sequential(
            conv_fn(num_features, out_channels), norm_fn(out_channels),
        )
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


class SRResNetDecoder(nn.Module):
    """'Decoder' part of SRResNet, converting embeddings to output image.

    It has been proposed in `Photo-Realistic Single Image Super-Resolution
    Using a Generative Adversarial Network`_.

    Args:
        in_channels: Number of channels in the input embedding.
        out_channels: Number of channels in the output image.
        scale_factor: Ratio between the size of the high-resolution image
            (output) and its low-resolution counterpart (input).
            In other words multiplier for spatial size.
        conv_fn: Convolutional layers parameters.
        activation_fn: Activation function to use.

    .. _`Photo-Realistic Single Image Super-Resolution Using a Generative
        Adversarial Network`: https://arxiv.org/pdf/1609.04802.pdf

    """

    @utils.process_fn_params
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        scale_factor: int = 2,
        conv_fn: ModuleParams = blocks.Conv2d,
        activation_fn: ModuleParams = nn.PReLU,
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
            upsampling_block = blocks.SubPixelConv(
                num_features=in_channels,
                conv_fn=conv_fn,
                activation_fn=activation_fn,
            )
            blocks_list.append((f"upsampling_{i}", upsampling_block))

        # highres conv
        last_conv = conv_fn(in_channels, out_channels)
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


__all__ = ["SRResNetEncoder", "SRResNetDecoder"]
