import torch
from torch import nn

from esrgan import utils

__all__ = ["EncoderDecoderNet"]


class EncoderDecoderNet(nn.Module):
    """Generalized Encoder-Decoder network.

    Args:
        encoder: Encoder module, usually used for the extraction
            of embeddings from input signals.
        decoder: Decoder module, usually used for embeddings processing
            e.g. generation of signal similar to the input one (in GANs).

    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # TODO:
        utils.net_init_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Args:
            x: Batch of input signals e.g. images.

        Returns:
            Batch of generated signals e.g. images.

        """
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.clamp(x, min=0.0, max=1.0)

        return x
