import copy
from typing import Optional

from catalyst.registry import MODULE
import torch
from torch import nn

from esrgan import utils


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

    @classmethod
    def get_from_params(
        cls,
        encoder_params: Optional[dict] = None,
        pooling_params: Optional[dict] = None,
        head_params: Optional[dict] = None,
    ) -> "VGGConv":
        """Create model based on it config.

        Args:
            encoder_params: Params of encoder module.
            pooling_params: Params of the pooling layer.
            head_params: 'Head' module params.

        Returns:
            Model.

        """
        encoder: nn.Module = nn.Identity()
        if (encoder_params_ := copy.deepcopy(encoder_params)) is not None:
            encoder_fn = MODULE.get(encoder_params_.pop("module"))
            encoder = encoder_fn(**encoder_params_)

        pool: nn.Module = nn.Identity()
        if (pooling_params_ := copy.deepcopy(pooling_params)) is not None:
            pool_fn = MODULE.get(pooling_params_.pop("module"))
            pool = pool_fn(**pooling_params_)

        head: nn.Module = nn.Identity()
        if (head_params_ := copy.deepcopy(head_params)) is not None:
            head_fn = MODULE.get(head_params_.pop("module"))
            head = head_fn(**head_params_)

        net = cls(encoder=encoder, pool=pool, head=head)
        utils.net_init_(net)

        return net


__all__ = ["VGGConv"]
