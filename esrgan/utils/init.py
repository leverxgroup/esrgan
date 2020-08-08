import copy
import inspect
import math
from typing import Any, Callable, Optional, Union

import torch
from torch import nn


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> None:
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification`_.

    Args:
        tensor: An n-dimensional tensor.
        a: The slope of the rectifier used after this layer
            (only used with ``'leaky_relu'`` and ``'prelu'``).
        mode: Either ``'fan_in'`` or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes
            in the backwards pass.
        nonlinearity: The non-linear function (`nn.functional` name).

    .. _`Delving deep into rectifiers: Surpassing human-level performance
        on ImageNet classification`: https://arxiv.org/pdf/1502.01852.pdf

    """
    base_act = "relu" if nonlinearity == "prely" else nonlinearity
    nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=base_act)

    if nonlinearity == "prelu":
        with torch.no_grad():
            std_correction = math.sqrt(1 + a ** 2)
            tensor.div_(std_correction)


def module_init_(
    module: nn.Module,
    nonlinearity: Union[str, nn.Module, None] = None,
    **kwargs: Any,
) -> None:
    """Initialize module based on the activation function.

    Args:
        module: Module to initialize.
        nonlinearity: Activation function. If LeakyReLU/PReLU and of type
            `nn.Module`, then initialization will be adapted by value of slope.
        **kwargs: Additional params to pass in init function.

    """
    # get name of activation function and extract slope param if possible
    activation_name: Optional[str] = None
    init_kwargs = copy.deepcopy(kwargs)
    if isinstance(nonlinearity, str):
        activation_name = nonlinearity.lower()
    elif isinstance(nonlinearity, nn.Module):
        activation_name = nonlinearity.__class__.__name__.lower()
        assert isinstance(activation_name, str)

        if activation_name == "leakyrelu":  # leakyrelu == LeakyReLU.lower
            activation_name = "leaky_relu"
            init_kwargs["a"] = kwargs.get("a", nonlinearity.negative_slope)
        elif activation_name == "prelu":
            init_kwargs["a"] = kwargs.get("a", nonlinearity.weight.data)

    # select initialization
    if activation_name in {"sigmoid", "tanh"}:
        weignt_init_fn: Callable = nn.init.xavier_uniform_
        init_kwargs = kwargs
    elif activation_name in {"relu", "elu", "leaky_relu", "prelu"}:
        weignt_init_fn = kaiming_normal_
        init_kwargs["nonlinearity"] = activation_name
    else:
        weignt_init_fn = nn.init.normal_
        init_kwargs["std"] = kwargs.get("std", 0.01)

    # init weights of the module
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        weignt_init_fn(module.weight, **init_kwargs)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def net_init_(net: nn.Module) -> None:
    """Inplace initialization of weights of neural network.

    Args:
        net: Network to initialize.

    """
    # create set of all activation functions (in PyTorch)
    activations = tuple(
        m[1]
        for m in inspect.getmembers(nn.modules.activation, inspect.isclass)
        if m[1].__module__ == "torch.nn.modules.activation"
    )

    # init of the layer depends on activation used after it,
    #  so iterate from the last layer to the first
    activation: Optional[nn.Module] = None
    for m in reversed(list(net.modules())):
        if isinstance(m, activations):
            activation = m

        module_init_(m, nonlinearity=activation)


__all__ = ["kaiming_normal_", "module_init_", "net_init_"]
