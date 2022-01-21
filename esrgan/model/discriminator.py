import torch
from torch import nn

from esrgan import utils

__all__ = ["VGGConv"]


class VGGConv(nn.Module):
    """VGG-like neural network for image classification.

    Args:
        encoder: Image encoder module, usually used for the extraction
            of embeddings from input signals.
        pool: Pooling layer, used to reduce embeddings from the encoder.
        head: Classification head, usually consists of Fully Connected layers.

    """

    def __init__(
        self, encoder: nn.Module, pool: nn.Module, head: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.pool = pool
        self.head = head

        # TODO:
        utils.net_init_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call.

        Args:
            x: Batch of images.

        Returns:
            Batch of logits.

        """
        x = self.pool(self.encoder(x))
        x = x.view(x.shape[0], -1)
        x = self.head(x)

        return x
