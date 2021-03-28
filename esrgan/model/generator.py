import copy
from typing import Optional

from catalyst.registry import REGISTRY
import torch
from torch import nn

from esrgan import utils


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

    @classmethod
    def get_from_params(
        cls,
        encoder_params: Optional[dict] = None,
        decoder_params: Optional[dict] = None,
    ) -> "EncoderDecoderNet":
        """Create model based on it config.

        Args:
            encoder_params: Encoder module params.
            decoder_params: Decoder module parameters.

        Returns:
            Model.

        """
        encoder: nn.Module = nn.Identity()
        if (encoder_params_ := copy.deepcopy(encoder_params)) is not None:
            encoder_fn = REGISTRY.get(encoder_params_.pop("module"))
            encoder = encoder_fn(**encoder_params_)

        decoder: nn.Module = nn.Identity()
        if (decoder_params_ := copy.deepcopy(decoder_params)) is not None:
            decoder_fn = REGISTRY.get(decoder_params_.pop("module"))
            decoder = decoder_fn(**decoder_params_)

        net = cls(encoder=encoder, decoder=decoder)
        utils.net_init_(net)

        return net


__all__ = ["EncoderDecoderNet"]
