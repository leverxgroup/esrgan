from typing import Any, Callable, Optional

from torch import nn

__all__ = ["create_layer"]


def create_layer(
    layer: Callable[..., nn.Module],
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    layer_name: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Helper function to generalize layer creation.

    Args:
        layer: Layer constructor.
        in_channels: Size of the input sample.
        out_channels: Size of the output e.g. number of channels
            produced by the convolution.
        layer_name: Name of the layer e.g. ``'activation'``.
        **kwargs: Additional params to pass into `layer` function.

    Returns:
        Layer.

    Examples:
        >>> in_channels, out_channels = 10, 5
        >>> create_layer(nn.Linear, in_channels, out_channels)
        Linear(in_features=10, out_features=5, bias=True)
        >>> create_layer(nn.ReLU, in_channels, out_channels, layer_name='act')
        ReLU()

    """
    module: nn.Module
    if layer_name in {"activation", "act", "dropout", "pool", "pooling"}:
        module = layer(**kwargs)
    elif layer_name in {"normalization", "norm", "bn"}:
        module = layer(out_channels, **kwargs)
    else:
        module = layer(in_channels, out_channels, **kwargs)

    return module
