from typing import Dict

from catalyst import dl
import torch


class GANRunner(dl.Runner):
    """Runner for experiments with supervised / GAN model."""

    def _init(
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
        super()._init()

        self.generator_key = generator_key
        self.discriminator_key = discriminator_key

        self.input_key = input_key
        self.target_key = target_key
        self.generator_output_key = generator_output_key
        self.discriminator_real_output_gkey = discriminator_real_output_gkey
        self.discriminator_fake_output_gkey = discriminator_fake_output_gkey
        self.discriminator_real_output_dkey = discriminator_real_output_dkey
        self.discriminator_fake_output_dkey = discriminator_fake_output_dkey

    def _prepare_for_stage(self, stage: str) -> None:
        """Prepare `_handle_batch` method for current stage.

        Args:
            stage: Name of current stage.

        Raises:
            NotImplementedError: Name of the `stage` should ends with
                ``'_supervised'``, ``'_gan'`` or should be ``'infer'``,
                raise error otherwise.

        """
        super()._prepare_for_stage(stage=stage)

        if stage.endswith("_supervised") or stage == "infer":
            self._handle_batch = self._handle_batch_supervised
        elif stage.endswith("_gan"):
            self._handle_batch = self._handle_batch_gan
        else:
            raise NotImplementedError()

    def _handle_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        # `_handle_batch` method is @abstractmethod so it must be defined
        # even if it overwrites in `_prepare_for_stage`
        raise NotImplementedError()

    def _handle_batch_supervised(self, batch: Dict[str, torch.Tensor]) -> None:
        """Process train/valid batch, supervised mode.

        Args:
            batch: Input batch (batch of samples).

        """
        model = self.model[self.generator_key]
        output = model(batch[self.input_key])

        self.output = {self.generator_output_key: output}

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

        noise = torch.randn(real_image.shape, device=self.device)
        real_image = torch.clamp((real_image + 0.05 * noise), min=0.0, max=1.0)

        # predictions used in calculation of adversarial loss of generator
        real_logits_g = discriminator(real_image)
        fake_logits_g = discriminator(fake_image)

        # predictions used in calculation of adversarial loss of discriminator
        real_logits_d = discriminator(real_image)
        fake_logits_d = discriminator(fake_image.detach())

        self.output = {
            self.generator_output_key: fake_image,
            self.discriminator_real_output_gkey: real_logits_g,
            self.discriminator_fake_output_gkey: fake_logits_g,
            self.discriminator_real_output_dkey: real_logits_d,
            self.discriminator_fake_output_dkey: fake_logits_d,
        }

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


__all__ = ["GANRunner"]
