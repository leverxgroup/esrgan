import collections
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from esrgan import utils
from esrgan.model.module import blocks
from esrgan.utils.types import ModuleParams


class StridedConvEncoder(nn.Module):
    """Generalized Fully Convolutional encoder.

    Args:
        layers: List of feature maps sizes of each block.
        layer_order: Ordered list of layers applied within each block.
            For instance, if you don't want to use normalization layer
            just exclude it from this list.
        conv_fn: Convolutional layer params.
        activation_fn: Activation function to use.
        norm_fn: Normalization layer params, e.g. :py:class:`.nn.BatchNorm2d`.
        residual_fn: Block wrapper function, e.g.
            :py:class:`~.blocks.container.ResidualModule` can be used
            to add residual connections between blocks.

    """

    @utils.process_fn_params
    def __init__(
        self,
        layers: Iterable[int] = (3, 64, 128, 128, 256, 256, 512, 512),
        layer_order: Iterable[str] = ("conv", "norm", "activation"),
        conv_fn: ModuleParams = blocks.Conv2d,
        activation_fn: ModuleParams = blocks.LeakyReLU,
        norm_fn: Optional[ModuleParams] = nn.BatchNorm2d,
        residual_fn: Optional[ModuleParams] = None,
    ):
        super().__init__()

        name2fn: Dict[str, Callable[..., nn.Module]] = {
            "activation": activation_fn,
            "conv": conv_fn,
            "norm": norm_fn,
        }

        self._layers = list(layers)

        net: List[Tuple[str, nn.Module]] = []

        first_conv = collections.OrderedDict([
            ("conv_0", name2fn["conv"](self._layers[0], self._layers[1])),
            ("act", name2fn["activation"]()),
        ])
        net.append(("block_0", nn.Sequential(first_conv)))

        channels = utils.pairwise(self._layers[1:])
        for i, (in_ch, out_ch) in enumerate(channels, start=1):
            block_list: List[Tuple[str, nn.Module]] = []
            for name in layer_order:
                # `conv + 2x2 pooling` is equal to `conv with stride=2`
                kwargs = {"stride": out_ch // in_ch} if name == "conv" else {}

                module = utils.create_layer(
                    layer_name=name,
                    layer=name2fn[name],
                    in_channels=in_ch,
                    out_channels=out_ch,
                    **kwargs
                )
                block_list.append((name, module))
            block = nn.Sequential(collections.OrderedDict(block_list))

            # add residual connection, like in resnet blocks
            if residual_fn is not None and in_ch == out_ch:
                block = residual_fn(block)

            net.append((f"block_{i}", block))

        self.net = nn.Sequential(collections.OrderedDict(net))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.net(x)

        return output

    @property
    def in_channels(self) -> int:
        """The number of channels in the feature map of the input."""
        return self._layers[0]

    @property
    def out_channels(self) -> int:
        """Number of channels produced by the block."""
        return self._layers[-1]


__all__ = ["StridedConvEncoder"]
