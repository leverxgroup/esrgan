from typing import Iterable

import torch
from torch import nn

__all__ = ["ConcatInputModule", "ResidualModule"]


class ConcatInputModule(nn.Module):
    """Module wrapper, passing outputs of all previous layers
    into each next layer.

    Args:
        module: PyTorch layer to wrap.

    """

    def __init__(self, module: Iterable[nn.Module]) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.

        """
        output = [x]
        for module in self.module:
            z = torch.cat(output, dim=1)
            output.append(module(z))

        return output[-1]


class ResidualModule(nn.Module):
    """Residual wrapper, adds identity connection.

    It has been proposed in `Deep Residual Learning for Image Recognition`_.

    Args:
        module: PyTorch layer to wrap.
        scale: Residual connections scaling factor.
        requires_grad: If set to ``False``, the layer will not learn
            the strength of the residual connection.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/pdf/1512.03385.pdf

    """

    def __init__(
        self,
        module: nn.Module,
        scale: float = 1.0,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.module = module
        self.scale = nn.Parameter(
            torch.tensor(scale), requires_grad=requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.

        """
        return x + self.scale * self.module(x)
