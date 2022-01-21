from typing import Any, Callable, Dict, Optional

__all__ = ["Augmentor"]


class Augmentor:
    """Applies provided transformation on dictionaries."""

    def __init__(
        self, transform: Optional[Callable[[Any], Dict]] = None
    ) -> None:
        """Constructor method for the :py:class:`Augmentor` class.

        Args:
            transform: A function / transform that takes in dictionary
            and returns a transformed version.
            If ``None``, the identity function is used.

        """
        self.transform = transform if transform is not None else self.indentity

    def __call__(self, dict_: Dict) -> Dict:
        return self.transform(**dict_)

    @staticmethod
    def indentity(dict_: Dict) -> Dict:
        return dict_
