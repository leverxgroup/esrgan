from catalyst.dl import registry
from esrgan.core import SRExperiment as Experiment, GANRunner as Runner
from esrgan import callbacks, criterions, model


for registry_key, module in zip(
    ("CALLBACK", "CRITERION", "MODULE", "MODEL"),
    (callbacks, criterions, model.module, model),
):
    registry.__dict__[registry_key].add_from_module(module, prefix=["esrgan."])
