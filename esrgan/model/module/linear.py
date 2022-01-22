from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from esrgan import utils
from esrgan.model.module import blocks

__all__ = ["LinearHead"]


class LinearHead(nn.Module):
    """Stack of linear layers used for embeddings classification.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        latent_channels: Size of the latent space.
        layer_order: Ordered list of layers applied within each block.
            For instance, if you don't want to use activation function
            just exclude it from this list.
        linear: Class constructor or partial object which when called
            should return linear layer e.g., :py:class:`nn.Linear`.
        activation: Class constructor or partial object which when called
            should return activation function layer e.g., :py:class:`nn.ReLU`.
        norm: Class constructor or partial object which when called
            should return normalization layer e.g., :py:class:`nn.BatchNorm1d`.
        dropout: Class constructor or partial object which when called
            should return dropout layer e.g., :py:class:`nn.Dropout`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: Optional[Iterable[int]] = None,
        layer_order: Iterable[str] = ("linear", "activation"),
        linear: Callable[..., nn.Module] = nn.Linear,
        activation: Callable[..., nn.Module] = blocks.LeakyReLU,
        norm: Optional[Callable[..., nn.Module]] = None,
        dropout: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        name2fn: Dict[str, Callable[..., nn.Module]] = {
            "activation": activation,
            "dropout": dropout,
            "linear": linear,
            "norm": norm,
        }

        latent_channels = latent_channels or []
        channels = [in_channels, *latent_channels, out_channels]
        channels_pairs: List[Tuple[int, int]] = list(utils.pairwise(channels))

        net: List[nn.Module] = []
        for in_ch, out_ch in channels_pairs[:-1]:
            for name in layer_order:
                module = utils.create_layer(
                    layer_name=name,
                    layer=name2fn[name],
                    in_channels=in_ch,
                    out_channels=out_ch,
                )
                net.append(module)
        net.append(name2fn["linear"](*channels_pairs[-1]))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs e.g. images.

        Returns:
            Batch of logits.

        """
        output = self.net(x)

        return output
