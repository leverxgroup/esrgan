from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from esrgan import utils
from esrgan.model.module import blocks
from esrgan.utils.types import ModuleParams


class LinearHead(nn.Module):
    """Stack of linear layers used for embeddings classification.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        latent_channels: Size of the latent space.
        layer_order: Ordered list of layers applied within each block.
            For instance, if you don't want to use normalization layer
            just exclude it from this list.
        linear_fn: Linear layer params.
        activation_fn: Activation function to use.
        norm_fn: Normalization layer params, e.g. :py:class:`nn.BatchNorm1d`.
        dropout_fn: Dropout layer params, e.g. :py:class:`nn.Dropout`.

    """

    @utils.process_fn_params
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: Optional[Iterable[int]] = None,
        layer_order: Iterable[str] = ("linear", "activation"),
        linear_fn: ModuleParams = nn.Linear,
        activation_fn: ModuleParams = blocks.LeakyReLU,
        norm_fn: Optional[ModuleParams] = None,
        dropout_fn: Optional[ModuleParams] = None,
    ) -> None:
        super().__init__()

        name2fn: Dict[str, Callable[..., nn.Module]] = {
            "activation": activation_fn,
            "dropout": dropout_fn,
            "linear": linear_fn,
            "norm": norm_fn,
        }

        latent_channels = latent_channels if latent_channels else []
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


__all__ = ["LinearHead"]
