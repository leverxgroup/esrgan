import collections
from typing import Any, List, Tuple

from torch import nn

from esrgan import utils
from esrgan.model.module.blocks import container, Conv2d, LeakyReLU
from esrgan.utils.types import ModuleParams


class ResidualDenseBlock(container.ResidualModule):
    """Basic block of :py:class:`ResidualInResidualDenseBlock`.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_blocks: Number of convolutional blocks to use to form dense block.
        conv_fn: Convolutional layers parameters.
        activation_fn: Activation function to use after each conv layer.
        residual_scaling: Residual connections scaling factor.

    """

    @utils.process_fn_params
    def __init__(
        self,
        num_features: int,
        growth_channels: int,
        num_blocks: int = 5,
        conv_fn: ModuleParams = Conv2d,
        activation_fn: ModuleParams = LeakyReLU,
        residual_scaling: float = 0.2,
    ) -> None:
        in_channels = [
            num_features + i * growth_channels for i in range(num_blocks)
        ]
        out_channels = [growth_channels] * (num_blocks - 1) + [num_features]

        blocks: List[nn.Module] = []
        for in_channels_, out_channels_ in zip(in_channels, out_channels):
            block = collections.OrderedDict([
                ("conv", conv_fn(in_channels_, out_channels_)),
                ("act", activation_fn()),
            ])
            blocks.append(nn.Sequential(block))

        super().__init__(
            module=container.ConcatInputModule(nn.ModuleList(blocks)),
            scale=residual_scaling,
        )


class ResidualInResidualDenseBlock(container.ResidualModule):
    """Residual-in-Residual Dense Block (RRDB).

    Look at the paper: `ESRGAN: Enhanced Super-Resolution Generative
    Adversarial Networks`_ for more details.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_dense_blocks: Number of dense blocks to use to form `RRDB` block.
        residual_scaling: Residual connections scaling factor.
        **kwargs: Dense block params.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        num_dense_blocks: int = 3,
        residual_scaling: float = 0.2,
        **kwargs: Any,
    ) -> None:
        blocks: List[Tuple[str, nn.Module]] = []
        for i in range(num_dense_blocks):
            block = ResidualDenseBlock(
                num_features=num_features,
                growth_channels=growth_channels,
                residual_scaling=residual_scaling,
                **kwargs,
            )
            blocks.append((f"block_{i}", block))

        super().__init__(
            module=nn.Sequential(collections.OrderedDict(blocks)),
            scale=residual_scaling
        )


__all__ = ["ResidualDenseBlock", "ResidualInResidualDenseBlock"]
