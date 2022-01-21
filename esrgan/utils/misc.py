import itertools
from typing import Any, Iterable

__all__ = ["pairwise", "is_power_of_two"]


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """Iterate sequences by pairs.

    Args:
        iterable: Any iterable sequence.

    Returns:
        Pairwise iterator.

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        ...     print(i)
        (1, 2)
        (2, 5)
        (5, -3)

    """
    a, b = itertools.tee(iterable, 2)
    next(b, None)
    return zip(a, b)


def is_power_of_two(number: int) -> bool:
    """Check if a given number is a power of two.

    Args:
        number: Nonnegative integer.

    Returns:
        ``True`` or ``False``.

    Examples:
        >>> is_power_of_two(4)
        True
        >>> is_power_of_two(3)
        False

    """
    result = number == 0 or (number & (number - 1) != 0)
    return result
