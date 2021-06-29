import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class AdversarialLoss(_Loss):
    """GAN Loss function.

    Args:
        mode: Specifies loss terget: ``'generator'`` or ``'discriminator'``.
            ``'generator'``: maximize probability that fake data drawn from
            real data distribution (it is useful when training generator),
            ``'discriminator'``: minimize probability that real and generated
            distributions are similar.

    Raises:
        NotImplementedError: If `mode` not ``'generator'``
                or ``'discriminator'``.

    """

    def __init__(self, mode: str = "discriminator") -> None:
        super().__init__()

        if mode == "generator":
            self.forward = self.forward_generator
        elif mode == "discriminator":
            self.forward = self.forward_discriminator
        else:
            raise NotImplementedError()

    def forward_generator(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (generator mode).

        Args:
            fake_logits: Predictions of discriminator for fake data.
            real_logits: Predictions of discriminator for real data.

        Returns:
            Loss, scalar.

        """
        loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )

        return loss

    def forward_discriminator(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (discriminator mode).

        Args:
            fake_logits: Predictions of discriminator for fake data.
            real_logits: Predictions of discriminator for real data.

        Returns:
            Loss, scalar.

        """
        loss_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits), reduction="sum"
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits), reduction="sum"
        )

        num_samples = real_logits.shape[0] + fake_logits.shape[0]
        loss = (loss_real + loss_fake) / num_samples  # mean loss

        return loss


class RelativisticAdversarialLoss(_Loss):
    """Relativistic average GAN loss function.

    It has been proposed in `The relativistic discriminator: a key element
    missing from standard GAN`_.

    Args:
        mode: Specifies loss target: ``'generator'`` or ``'discriminator'``.
            ``'generator'``: maximize probability that fake data more realistic
            than real (it is useful when training generator),
            ``'discriminator'``: maximize probability that real data more
            realistic than fake (useful when training discriminator).

    Raises:
        NotImplementedError: If `mode` not ``'generator'``
            or ``'discriminator'``.

    .. _`The relativistic discriminator: a key element missing
        from standard GAN`: https://arxiv.org/pdf/1807.00734.pdf

    """

    def __init__(self, mode: str = "discriminator") -> None:
        super().__init__()

        if mode == "generator":
            self.rf_labels, self.fr_labels = 0, 1
        elif mode == "discriminator":
            self.rf_labels, self.fr_labels = 1, 0
        else:
            raise NotImplementedError()

    def forward(
        # self, outputs: torch.Tensor, targets: torch.Tensor
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation method for the relativistic adversarial loss.

        Args:
            fake_logits: Probability that generated samples are not real.
            real_logits: Probability that real (ground truth) samples are fake.

        Returns:
            Loss, scalar.

        """
        loss_rf = F.binary_cross_entropy_with_logits(
            input=(real_logits - fake_logits.mean()),
            target=torch.empty_like(real_logits).fill_(self.rf_labels),
        )
        loss_fr = F.binary_cross_entropy_with_logits(
            input=(fake_logits - real_logits.mean()),
            target=torch.empty_like(fake_logits).fill_(self.fr_labels),
        )
        loss = (loss_fr + loss_rf) / 2

        return loss


__all__ = ["AdversarialLoss", "RelativisticAdversarialLoss"]
