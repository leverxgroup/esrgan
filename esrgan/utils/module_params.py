import copy
import functools
import re
from typing import Any, Callable, Dict, Optional

from catalyst.registry import REGISTRY
from torch import nn

from esrgan.utils.types import ModuleParams


def process_fn_params(function: Callable) -> Callable:
    """Decorator for `fn_params` processing.

    Decorator that process all `*_fn` parameters and replaces ``str`` and
    ``dict`` values with corresponding constructors of `nn` modules.
    For example for ``act_fn='ReLU'`` and ``act_fn=nn.ReLU`` parameters
    the result will be ``nn.ReLU`` constructor of ReLU activation function,
    and for ``act_fn={'act': 'ReLU', 'inplace': True}`` the result
    will be 'partial' constructor ``nn.ReLU`` in which
    ``inplace`` argument is set to ``True``.

    Args:
        function: Function to wrap.

    Returns:
        Wrapped function.

    """
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        kwargs_: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if (match := re.match(r"(\w+)_fn", key)) and value:
                value = _process_fn_params(
                    params=value, key=match.group(1)
                )
            kwargs_[key] = value

        output = function(*args, **kwargs_)

        return output
    return wrapper


def _process_fn_params(
    params: ModuleParams, key: Optional[str] = None
) -> Callable[..., nn.Module]:
    module_fn: Callable[..., nn.Module]
    if callable(params):
        module_fn = params
    elif isinstance(params, str):
        name = params
        module_fn = REGISTRY.get(name)
    elif isinstance(params, dict) and key is not None:
        params = copy.deepcopy(params)

        name_or_fn = params.pop(key)
        module_fn = _process_fn_params(name_or_fn)
        module_fn = functools.partial(module_fn, **params)
    else:
        NotImplementedError()

    return module_fn


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


__all__ = ["process_fn_params", "create_layer"]
