from typing import Any, Callable, Dict, Optional

__all__ = ["Augmentor"]


def indentity(d: Dict) -> Dict:
    """A placeholder identity operator that is argument-insensitive.

    Args:
        d: Dictionary with the data that describes sample.

    Returns:
        Same dictionary ``d``.

    """
    return d


class Augmentor:
    """Applies provided transformation on dictionaries.

    Args:
        transform: A function / transform that takes in dictionary
            and returns a transformed version.
            If ``None``, the identity function is used.

    """

    def __init__(
        self, transform: Optional[Callable[[Any], Dict]] = None
    ) -> None:
        self.transform = transform if transform is not None else indentity

    def __call__(self, d: Dict) -> Dict:
        """Applies ``transform`` to the dictionary ``d``.

        Args:
            d: Dictionary to transform.

        Returns:
            Output of the ``transform`` function.

        """
        return self.transform(**d)
