from catalyst.registry import REGISTRY
from catalyst.tools.registry import Registry

from esrgan import callbacks, criterions, dataset, model, runner


# TODO: remove this hotfix (for catalyst=21.03)
def _transforms_loader(r: Registry):
    import albumentations as albu
    from albumentations import pytorch as albu_torch

    r.add_from_module(albu, prefix=["A.", "albu.", "albumentations."])
    r.add_from_module(albu_torch, prefix=["A.", "albu.", "albumentations."])


REGISTRY.late_add(_transforms_loader)


for module in (callbacks, criterions, dataset, model.module, model, runner):
    REGISTRY.add_from_module(module, prefix=["esrgan."])
