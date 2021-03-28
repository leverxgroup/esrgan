import copy
from typing import Callable, Dict, Optional

from catalyst import runners
from catalyst.registry import REGISTRY
from torch.utils.data import Dataset

from esrgan.runner import gan_runner


def get_transform_from_params(**params) -> Optional[Callable]:
    if "transforms" in params:
        transforms_composition = [
            get_transform_from_params(**transform_params)
            for transform_params in params["transforms"]
        ]
        params.update(transforms=transforms_composition)

    transform = REGISTRY.get_from_params(**params)

    return transform


class GANConfigRunner(runners.ConfigRunner, gan_runner.GANRunner):
    def __init__(
        self,
        config: dict,
        input_key: str = "image",
        target_key: str = "real_image",
        generator_output_key: str = "fake_image",
        discriminator_real_output_gkey: str = "g_real_logits",
        discriminator_fake_output_gkey: str = "g_fake_logits",
        discriminator_real_output_dkey: str = "d_real_logits",
        discriminator_fake_output_dkey: str = "d_fake_logits",
        generator_key: str = "generator",
        discriminator_key: str = "discriminator",
    ):
        gan_runner.GANRunner.__init__(
            self,
            input_key=input_key,
            target_key=target_key,
            generator_output_key=generator_output_key,
            discriminator_real_output_gkey=discriminator_real_output_gkey,
            discriminator_fake_output_gkey=discriminator_fake_output_gkey,
            discriminator_real_output_dkey=discriminator_real_output_dkey,
            discriminator_fake_output_dkey=discriminator_fake_output_dkey,
            generator_key=generator_key,
            discriminator_key=discriminator_key,
        )

        runners.ConfigRunner.__init__(self, config=config)

    def get_transforms(self, **params) -> Callable[[dict], dict]:
        """Returns transform for a given stage and dataset.

        Args:
            stage (str): stage name
            dataset (str): dataset name (e.g. "train", "valid"),
                will be used only if the value of `_key_value`` is ``True``

        Returns:
            Callable: transform function

        """
        transform = get_transform_from_params(**params)

        if transform is not None:
            def transform_fn(dict_):
                return transform(**dict_)
        else:
            def transform_fn(dict_):
                return dict_

        return transform_fn

    def get_datasets(
        self,
        stage: str,
        train_dataset: Optional[Dict] = None,
        valid_dataset: Optional[Dict] = None,
        infer_dataset: Optional[Dict] = None,
    ) -> Dict[str, Dataset]:
        """Returns the datasets for a given stage and epoch.

        Args:
            stage: stage name of interest, e.g. "train", "finetune", "gan" ...
            train_dataset: Parameters of train dataset, must contain
                ``'dataset'`` key with the name of dataset to use
                e.g. :py:class:`esrgan.dataset.DIV2KDataset`.
            valid_dataset: Parameters of validation dataset.
            infer_dataset: Parameters of inference dataset.

        Returns:
            Dictionary with datasets for current stage.

        Example for Config API:

        .. code-block:: yaml

            train_dataset:
              dataset: DIV2KDataset
              root: data
              train: true
              target_type: bicubic_X4
              download: true

        """
        datasets: Dict[str, Dataset] = {}
        for params, mode in zip(
            (train_dataset, valid_dataset, infer_dataset),
            ("train", "valid", "infer"),
        ):
            if params:
                dataset_params = copy.deepcopy(params)

                transform_params = dataset_params.pop('transform', {})
                transform = self.get_transforms(**transform_params)

                datasets[mode] = REGISTRY.get_from_params(
                    **dataset_params, transform=transform
                )

        return datasets


__all__ = ["GANConfigRunner"]
