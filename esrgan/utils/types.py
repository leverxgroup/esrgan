from typing import Any, Callable, Dict, Union

from torch import nn

ModuleParams = Union[Callable[..., nn.Module], str, Dict[str, Any]]
