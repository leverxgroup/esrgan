import functools
from typing import Callable, Union

from catalyst import callbacks, utils
import piq
import torch


# TODO: remove this hotfix (for catalyst=21.03)
def to_numpy(metric_fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        value = metric_fn(*args, **kwargs)
        if torch.is_tensor(value):
            value = utils.detach_tensor(value)

        return value
    return wrapper


class PSNRCallback(callbacks.FunctionalMetricCallback):
    """Peak signal-to-noise ratio (PSNR) metric callback.

    Compute Peak Signal-to-Noise Ratio for a batch of images.

    Args:
        input_key: Input key to use for PSNR calculation;
            specifies our `y_true`.
        target_key: Output key to use for PSNR calculation;
            specifies our `y_pred`.
        metric_key: Name of the metric / key to store in logs.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Reduction over samples in batch, should be one of:
            ``'mean'``, ``'sum'``, or ``'none'``.
        convert_to_greyscale: If ``True``, convert RGB image to YCbCr format
            and computes PSNR only on luminance channel,
            compute on all 3 channels otherwise.

    """

    def __init__(
        self,
        input_key: str = "targets",
        target_key: str = "outputs",
        metric_key: str = "psnr",
        data_range: Union[int, float] = 1.0,
        reduction: str = "mean",
        convert_to_greyscale: bool = False,
    ) -> None:
        metric_fn = functools.partial(
            to_numpy(piq.psnr),
            data_range=data_range,
            reduction=reduction,
            convert_to_greyscale=convert_to_greyscale,
        )

        super().__init__(
            metric_fn=metric_fn,
            input_key=input_key,
            target_key=target_key,
            metric_key=metric_key,
        )


class SSIMCallback(callbacks.FunctionalMetricCallback):
    """Structural similarity (SSIM) metric callback.

    Computes Structural Similarity (SSIM) index between two images.
    It has been proposed in `Image Quality Assessment: From Error Visibility
    to Structural Similarity`__.

    Args:
        input_key: Input key to use for SSIM calculation;
            specifies our `y_true`.
        target_key: Output key to use for SSIM calculation;
            specifies our `y_pred`.
        metric_key: Name of the metric / key to store in logs.
        kernel_size: The side-length of the Gaussian sliding window
            used in comparison. Must be an odd value.
        kernel_sigma: Standard deviation of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output, should be
            one of: ``'mean'``, ``'sum'``, or ``'none'``.
        k1: Algorithm parameter, small constant used to stabilize the division
            with small denominator (see original paper for more info).
        k2: Algorithm parameter, small constant used to stabilize the division
            with small denominator (see original paper for more info).

    __ https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf

    """

    def __init__(
        self,
        input_key: str = "targets",
        target_key: str = "outputs",
        metric_key: str = "ssim",
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        data_range: Union[int, float] = 1.0,
        reduction: str = "mean",
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        metric_fn = functools.partial(
            to_numpy(piq.ssim),
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            data_range=data_range,
            reduction=reduction,
            k1=k1,
            k2=k2,
        )

        super().__init__(
            metric_fn=metric_fn,
            input_key=input_key,
            target_key=target_key,
            metric_key=metric_key,
        )


__all__ = ["PSNRCallback", "SSIMCallback"]
