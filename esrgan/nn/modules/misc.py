import functools
from typing import Callable, Tuple, Union

from torch import nn
from torch.nn.utils.spectral_norm import SpectralNorm

__all__ = ["Conv2d", "Conv2dSN", "LeakyReLU", "LinearSN"]


Conv2d: Callable[..., nn.Module] = functools.partial(
    nn.Conv2d, kernel_size=(3, 3), padding=1
)
LeakyReLU: Callable[..., nn.Module] = functools.partial(
    nn.LeakyReLU, negative_slope=0.2, inplace=True
)


class Conv2dSN(nn.Conv2d):
    """:py:class:`nn.Conv2d` + spectral normalization.

    Applies a 2D convolution over an input signal composed of several input
    planes. In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\\text{in}}, H, W)` and output
    :math:`(N, C_{\\text{out}}, H_{\\text{out}}, W_{\\text{out}})`
    can be precisely described as:

    .. math::
        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +
        \\sum_{k = 0}^{C_{\\text{in}} - 1} \\text{weight}(C_{\\text{out}_j}, k)
        \\star \\text{input}(N_i, k)

    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\\sigma` of the weight matrix calculated using
    power iteration method. See `Spectral Normalization for
    Generative Adversarial Networks`_  for details.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input
            channels to output channels.
        bias: If ``True``, adds a learnable bias to the output.
        padding_mode: ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``.
        n_power_iterations: Number of power iterations
            to calculate spectral norm.

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _`Spectral Normalization for Generative Adversarial Networks`:
        https://arxiv.org/abs/1802.05957

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        n_power_iterations: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        SpectralNorm.apply(
            module=self,
            n_power_iterations=n_power_iterations,
            name="weight",
            dim=0,
            eps=1e-12
        )


class LinearSN(nn.Linear):
    """:py:class:`nn.Linear` + spectral normalization.

    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\\sigma` of the weight matrix calculated using
    power iteration method. See `Spectral Normalization for
    Generative Adversarial Networks`_  for details.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
        n_power_iterations: Number of power iterations
            to calculate spectral norm.

    .. _`Spectral Normalization for Generative Adversarial Networks`:
        https://arxiv.org/abs/1802.05957

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_power_iterations: int = 1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        SpectralNorm.apply(
            module=self,
            n_power_iterations=n_power_iterations,
            name="weight",
            dim=0,
            eps=1e-12
        )
