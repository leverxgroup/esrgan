import copy
from typing import Dict, Optional

from catalyst import dl

from esrgan import dataset


class SRExperiment(dl.ConfigExperiment):
    """Experiment for ESRGAN, please check `catalyst docs`__ for more info.

    __ https://catalyst-team.github.io/catalyst/api/core.html#experiment

    """

    def get_datasets(
        self,
        stage: str,
        train_dataset_params: Optional[Dict] = None,
        valid_dataset_params: Optional[Dict] = None,
        infer_dataset_params: Optional[Dict] = None,
    ) -> dict:
        """Returns the datasets for a given stage and epoch.

        Args:
            stage: stage name of interest, e.g. "train", "finetune", "gan" ...
            train_dataset_params: Parameters of train dataset, must contain
                ``'dataset'`` key with the name of dataset to use
                e.g. :py:class:`esrgan.dataset.DIV2KDataset`.
            valid_dataset_params: Parameters of validation dataset.
            infer_dataset_params: Parameters of inference dataset.

        Returns:
            Dictionary with datasets for current stage.

        Example for Config API:

        .. code-block:: yaml

            train_dataset_params:
              dataset: DIV2KDataset
              root: data
              train: true
              target_type: bicubic_X4
              download: true

        """
        train_dataset_params = train_dataset_params or {}
        valid_dataset_params = valid_dataset_params or {}
        infer_dataset_params = infer_dataset_params or {}

        datasets = {}
        for params, mode in zip(
            (train_dataset_params, valid_dataset_params, infer_dataset_params),
            ("train", "valid", "infer"),
        ):
            if params:
                params_ = copy.deepcopy(params)

                dataset_name = params_.pop("dataset")
                dataset_ = dataset.__dict__[dataset_name]

                transform = self.get_transforms(stage=stage, dataset=mode)
                if transform is not None:
                    params_["transform"] = transform

                datasets[mode] = dataset_(**params_)

        return datasets


__all__ = ["SRExperiment"]
