import copy
from typing import Dict

from catalyst import runners
from catalyst.core.runner import IRunner
from catalyst.registry import REGISTRY
import torch

__all__ = ["GANRunner", "GANConfigRunner"]


class GANRunner(IRunner):
    """Runner for ESRGAN, please check `catalyst docs`__ for more info.

    __ https://catalyst-team.github.io/catalyst/api/core.html#experiment

    """

    def __init__(
        self,
        input_key: str = "image",
        target_key: str = "real_image",
        generator_output_key: str = "fake_image",
        discriminator_real_output_gkey: str = "g_real_logits",
        discriminator_fake_output_gkey: str = "g_fake_logits",
        discriminator_real_output_dkey: str = "d_real_logits",
        discriminator_fake_output_dkey: str = "d_fake_logits",
        generator_key: str = "generator",
        discriminator_key: str = "discriminator",
    ) -> None:
        """Catalyst-specific helper method for `__init__`.

        Args:
            input_key: Key in batch dict mapping for model input.
            target_key: Key in batch dict mapping for target.
            generator_output_key: Key in output dict model output
                of the generator will be stored under.
            discriminator_real_output_gkey: Key to store predictions of
                discriminator for real inputs, contain gradients for generator.
            discriminator_fake_output_gkey: Key to store predictions of
                discriminator for predictions of generator,
                contain gradients for generator.
            discriminator_real_output_dkey: Key to store predictions of
                discriminator for real inputs,
                contain gradients for discriminator only.
            discriminator_fake_output_dkey: Key to store predictions of
                discriminator for items produced by generator,
                contain gradients for discriminator only.
            generator_key: Key in model dict mapping for generator model.
            discriminator_key: Key in model dict mapping for discriminator
                model (will be used in gan stages only).

        """
        super().__init__()

        self.generator_key = generator_key
        self.discriminator_key = discriminator_key

        self.input_key = input_key
        self.target_key = target_key
        self.generator_output_key = generator_output_key
        self.discriminator_real_output_gkey = discriminator_real_output_gkey
        self.discriminator_fake_output_gkey = discriminator_fake_output_gkey
        self.discriminator_real_output_dkey = discriminator_real_output_dkey
        self.discriminator_fake_output_dkey = discriminator_fake_output_dkey

        self.epoch_counter = -1  # TODO: remove

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate predictions based on input batch (generator inference).

        Args:
            batch: Input batch (batch of samples to adjust e.g. zoom).

        Returns:
            Batch of predictions of the generator.

        """
        model = self.model[self.generator_key]
        output = model(batch[self.input_key].to(self.device))

        return output

    def handle_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Inner method to handle specified data batch.

        Args:
            batch: Dictionary with data batches from DataLoader.

        """
        # `handle_batch` method is `@abstractmethod` so it must be defined
        # even if it overwrites in `on_stage_start`
        # self.batch = {**batch}
        self._handle_batch_supervised(batch=batch)

    def _handle_batch_supervised(self, batch: Dict[str, torch.Tensor]) -> None:
        """Process train/valid batch, supervised mode.

        Args:
            batch: Input batch (batch of samples).

        """
        model = self.model[self.generator_key]
        output = model(batch[self.input_key])

        self.batch = {**batch, self.generator_output_key: output}

        # TODO: remove {
        if self.global_epoch_step > self.epoch_counter:
            self.epoch_counter = self.global_epoch_step

            self.log_image(tag=f'real image 1th', image=tensor_to_ndimage(batch['real_image'][0, ...].detach().cpu()))
            self.log_image(tag=f'real image 4th', image=tensor_to_ndimage(batch['real_image'][3, ...].detach().cpu()))

            self.log_image(tag=f'fake image 1th', image=tensor_to_ndimage(output[0, ...].detach().cpu()))
            self.log_image(tag=f'fake image 4th', image=tensor_to_ndimage(output[3, ...].detach().cpu()))
        # TODO: } remove

    def _handle_batch_gan(self, batch: Dict[str, torch.Tensor]) -> None:
        """Process train/valid batch, GAN mode.

        Args:
            batch: Input batch, should raw samples for generator
                and ground truth samples for discriminator.

        """
        generator = self.model[self.generator_key]
        discriminator = self.model[self.discriminator_key]

        real_image = batch[self.target_key]
        fake_image = generator(batch[self.input_key])

        noise = torch.randn(real_image.shape, device=real_image.device)
        real_image = torch.clamp((real_image + 0.05 * noise), min=0.0, max=1.0)

        # predictions used in calculation of adversarial loss of generator
        real_logits_g = discriminator(real_image)
        fake_logits_g = discriminator(fake_image)

        # predictions used in calculation of adversarial loss of discriminator
        real_logits_d = discriminator(real_image)
        fake_logits_d = discriminator(fake_image.detach())

        self.batch = {
            **batch,
            self.generator_output_key: fake_image,
            self.discriminator_real_output_gkey: real_logits_g,
            self.discriminator_fake_output_gkey: fake_logits_g,
            self.discriminator_real_output_dkey: real_logits_d,
            self.discriminator_fake_output_dkey: fake_logits_d,
        }

        # TODO: remove {
        if self.global_epoch_step > self.epoch_counter:
            self.epoch_counter = self.global_epoch_step

            self.log_image(tag=f'real image 1th', image=tensor_to_ndimage(batch['real_image'][0, ...].detach().cpu()))
            self.log_image(tag=f'real image 4th', image=tensor_to_ndimage(batch['real_image'][3, ...].detach().cpu()))

            self.log_image(tag=f'fake image 1th', image=tensor_to_ndimage(fake_image[0, ...].detach().cpu()))
            self.log_image(tag=f'fake image 4th', image=tensor_to_ndimage(fake_image[3, ...].detach().cpu()))
        # TODO: } remove

    def on_stage_start(self, runner: IRunner) -> None:
        """Prepare `_handle_batch` method for current stage.

        Args:
            runner: Current runner.

        Raises:
            NotImplementedError: Name of the stage should ends with
                ``'_supervised'``, ``'_gan'`` or should be ``'infer'``,
                raise error otherwise.

        """
        super().on_stage_start(runner=runner)

        if self.stage_key.endswith("_supervised") or self.stage_key == "infer":
            self.handle_batch = self._handle_batch_supervised
        elif self.stage_key.endswith("_gan"):
            self.handle_batch = self._handle_batch_gan
        else:
            raise NotImplementedError()


class GANConfigRunner(runners.ConfigRunner, GANRunner):
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
        GANRunner.__init__(
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

    @staticmethod
    def _get_model_from_params(**params):
        params = copy.deepcopy(params)
        is_key_value = params.pop("_key_value", False)

        if is_key_value:
            model = {
                model_key: runners.ConfigRunner._get_model_from_params(
                    **model_params
                )
                for model_key, model_params in params.items()
            }
            # TODO: hotfix for DDP
            # model = nn.ModuleDict(model)
        else:
            model = REGISTRY.get_from_params(**params)
        return model

# TODO: remove {
import numpy as np
from typing import Tuple
def tensor_to_ndimage(
    images: torch.Tensor,
    denormalize: bool = True,
    mean: Tuple[float, float, float] = (0, 0, 0),
    std: Tuple[float, float, float] = (1, 1, 1),
    move_channels_dim: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert float image(s) with standard normalization to
    np.ndarray with [0..1] when dtype is np.float32 and [0..255]
    when dtype is `np.uint8`.
    Args:
        images: [B]xCxHxW float tensor
        denormalize: if True, multiply image(s) by std and add mean
        mean (Tuple[float, float, float]): per channel mean to add
        std (Tuple[float, float, float]): per channel std to multiply
        move_channels_dim: if True, convert tensor to [B]xHxWxC format
        dtype: result ndarray dtype. Only float32 and uint8 are supported
    Returns:
        [B]xHxWxC np.ndarray of dtype
    """
    if denormalize:
        has_batch_dim = len(images.shape) == 4

        mean = images.new_tensor(mean).view(
            *((1,) if has_batch_dim else ()), len(mean), 1, 1
        )
        std = images.new_tensor(std).view(
            *((1,) if has_batch_dim else ()), len(std), 1, 1
        )

        images = images * std + mean

    images = images.float().clamp(0, 1).numpy()

    if move_channels_dim:
        images = np.moveaxis(images, -3, -1)

    if dtype == np.uint8:
        images = (images * 255).round().astype(dtype)
    else:
        assert dtype == np.float32, "Only float32 and uint8 are supported"

    return images
# TODO: } remove
