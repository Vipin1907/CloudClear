"""
training/losses.py
==================
Loss functions and optimiser configuration for the CloudClear-NER
Pix2Pix training pipeline.

Components
----------
* Generator Loss   — L1 pixel reconstruction + adversarial loss (BCE).
* Discriminator Loss — BCE on real / fake predictions with smooth labels.
* create_optimizers  — Adam optimisers for generator and discriminator.

Reference
---------
Isola et al., "Image-to-Image Translation with Conditional Adversarial
Networks", CVPR 2017.  (λ = 100 is the paper's default weighting.)

Author : CloudClear-NER Team
Created: 2026-07-05
"""

import torch
import torch.nn as nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Module-level loss instances (created once, shared across calls)
# ---------------------------------------------------------------------------

_l1_criterion: nn.L1Loss = nn.L1Loss()
_bce_criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

# Pix2Pix default weighting for the L1 term in generator loss
LAMBDA_L1: float = 100.0

# Adam hyper-parameters recommended by the Pix2Pix paper
ADAM_LR: float = 0.0002
ADAM_BETAS: tuple[float, float] = (0.5, 0.999)


# ---------------------------------------------------------------------------
# Generator Loss
# ---------------------------------------------------------------------------

def compute_generator_loss(
    fake_prediction: torch.Tensor,
    generated_image: torch.Tensor,
    target_image: torch.Tensor,
    lambda_l1: float = LAMBDA_L1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the total generator loss used in Pix2Pix training.

    The generator is penalised in two ways:

    1. **Adversarial loss** — pushes the generator to produce outputs that
       the discriminator classifies as *real* (label = 1).
    2. **L1 reconstruction loss** — enforces per-pixel fidelity between the
       generated image and the ground-truth clean target, weighted by λ.

    Total loss:
        ``total = adversarial_loss + λ × l1_loss``

    Parameters
    ----------
    fake_prediction : torch.Tensor
        Raw logits output by the discriminator for the generated image.
        Shape: ``(B, 1, H', W')`` or ``(B,)``.
    generated_image : torch.Tensor
        Output of the generator (cloud-removed prediction).
        Shape: ``(B, C, H, W)``.
    target_image : torch.Tensor
        Ground-truth cloud-free satellite patch.
        Shape: ``(B, C, H, W)``.
    lambda_l1 : float, optional
        Weight applied to the L1 term. Defaults to 100 (Pix2Pix default).

    Returns
    -------
    total_loss : torch.Tensor
        Scalar — weighted sum of adversarial and L1 losses.
    adversarial_loss : torch.Tensor
        Scalar — BCE loss encouraging the discriminator to rate fakes as real.
    l1_loss : torch.Tensor
        Scalar — mean absolute error between generated and target images.
    """
    # Generator wants the discriminator to output 1 ("real") for fake images
    real_labels: torch.Tensor = torch.ones_like(fake_prediction)

    adversarial_loss: torch.Tensor = _bce_criterion(fake_prediction, real_labels)
    l1_loss: torch.Tensor = _l1_criterion(generated_image, target_image)

    total_loss: torch.Tensor = adversarial_loss + lambda_l1 * l1_loss

    return total_loss, adversarial_loss, l1_loss


# ---------------------------------------------------------------------------
# Discriminator Loss
# ---------------------------------------------------------------------------

def compute_discriminator_loss(
    real_prediction: torch.Tensor,
    fake_prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the discriminator loss for one training step.

    The discriminator is trained to:

    * Classify **real** images (clean satellite patches) as **1**.
    * Classify **fake** images (generator output) as **0**.

    Total loss:
        ``discriminator_loss = (real_loss + fake_loss) / 2``

    Averaging by 2 (standard Pix2Pix practice) slows discriminator learning
    relative to the generator, preventing premature convergence.

    Parameters
    ----------
    real_prediction : torch.Tensor
        Raw logits from the discriminator on real (clean) image pairs.
        Shape: ``(B, 1, H', W')`` or ``(B,)``.
    fake_prediction : torch.Tensor
        Raw logits from the discriminator on generated (fake) image pairs.
        Shape: same as ``real_prediction``.

    Returns
    -------
    discriminator_loss : torch.Tensor
        Scalar — averaged sum of real and fake BCE losses.
    real_loss : torch.Tensor
        Scalar — BCE loss on real images (target label = 1).
    fake_loss : torch.Tensor
        Scalar — BCE loss on fake images (target label = 0).
    """
    # Real images → target label = 1
    real_labels: torch.Tensor = torch.ones_like(real_prediction)
    real_loss: torch.Tensor = _bce_criterion(real_prediction, real_labels)

    # Fake images → target label = 0
    fake_labels: torch.Tensor = torch.zeros_like(fake_prediction)
    fake_loss: torch.Tensor = _bce_criterion(fake_prediction, fake_labels)

    # Average the two terms (Pix2Pix convention)
    discriminator_loss: torch.Tensor = (real_loss + fake_loss) * 0.5

    return discriminator_loss, real_loss, fake_loss


# ---------------------------------------------------------------------------
# Optimiser Factory
# ---------------------------------------------------------------------------

def create_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    lr: float = ADAM_LR,
    betas: tuple[float, float] = ADAM_BETAS,
) -> tuple[Adam, Adam]:
    """
    Create Adam optimisers for the generator and discriminator.

    Uses the hyper-parameters recommended in the Pix2Pix paper:
    ``lr=0.0002``, ``betas=(0.5, 0.999)``.

    Parameters
    ----------
    generator     : nn.Module
        The UNetGenerator (or any generator) whose parameters to optimise.
    discriminator : nn.Module
        The PatchGAN discriminator (or any discriminator) to optimise.
    lr    : float, optional
        Learning rate. Defaults to 0.0002.
    betas : tuple[float, float], optional
        Adam beta coefficients. Defaults to (0.5, 0.999).

    Returns
    -------
    generator_optimizer     : torch.optim.Adam
    discriminator_optimizer : torch.optim.Adam
    """
    generator_optimizer: Adam = Adam(
        generator.parameters(),
        lr=lr,
        betas=betas,
    )

    discriminator_optimizer: Adam = Adam(
        discriminator.parameters(),
        lr=lr,
        betas=betas,
    )

    return generator_optimizer, discriminator_optimizer


# ---------------------------------------------------------------------------
# Quick-start smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Validates all loss functions and the optimiser factory using dummy tensors.
    No model training occurs — only forward-pass tensor arithmetic.

    Expected output (values will vary per run):
        --- Generator Loss ---
        Total Generator Loss : X.XXXXXX
        Adversarial Loss     : X.XXXXXX
        L1 Loss              : X.XXXXXX

        --- Discriminator Loss ---
        Discriminator Loss   : X.XXXXXX
        Real Loss            : X.XXXXXX
        Fake Loss            : X.XXXXXX

        --- Optimisers ---
        Optimizer Created Successfully

        [OK] All checks passed.
    """

    print("\n" + "=" * 50)
    print("  CloudClear-NER  |  training/losses.py — smoke-test")
    print("=" * 50)

    # -----------------------------------------------------------------------
    # Dummy tensors — mimic PatchGAN / UNet shapes
    # -----------------------------------------------------------------------
    BATCH: int = 4
    IMG_C, IMG_H, IMG_W = 3, 256, 256   # image space
    DISC_H, DISC_W = 30, 30             # PatchGAN output spatial size

    # Discriminator logit outputs (raw, un-sigmoided)
    fake_pred:   torch.Tensor = torch.randn(BATCH, 1, DISC_H, DISC_W)
    real_pred:   torch.Tensor = torch.randn(BATCH, 1, DISC_H, DISC_W)

    # Generator output and ground-truth target (already in [-1, 1])
    generated:   torch.Tensor = torch.rand(BATCH, IMG_C, IMG_H, IMG_W) * 2 - 1
    target:      torch.Tensor = torch.rand(BATCH, IMG_C, IMG_H, IMG_W) * 2 - 1

    # -----------------------------------------------------------------------
    # Generator loss
    # -----------------------------------------------------------------------
    total_g, adv_loss, l1_loss = compute_generator_loss(
        fake_pred, generated, target
    )

    print("\n--- Generator Loss ---")
    print(f"  Total Generator Loss : {total_g.item():.6f}")
    print(f"  Adversarial Loss     : {adv_loss.item():.6f}")
    print(f"  L1 Loss              : {l1_loss.item():.6f}")

    assert total_g.shape == torch.Size([]), "Generator total loss must be a scalar!"
    assert adv_loss.shape == torch.Size([]), "Adversarial loss must be a scalar!"
    assert l1_loss.shape  == torch.Size([]), "L1 loss must be a scalar!"

    # -----------------------------------------------------------------------
    # Discriminator loss
    # -----------------------------------------------------------------------
    disc_loss, real_loss, fake_loss = compute_discriminator_loss(
        real_pred, fake_pred
    )

    print("\n--- Discriminator Loss ---")
    print(f"  Discriminator Loss   : {disc_loss.item():.6f}")
    print(f"  Real Loss            : {real_loss.item():.6f}")
    print(f"  Fake Loss            : {fake_loss.item():.6f}")

    assert disc_loss.shape == torch.Size([]), "Discriminator loss must be a scalar!"
    assert real_loss.shape == torch.Size([]), "Real loss must be a scalar!"
    assert fake_loss.shape == torch.Size([]), "Fake loss must be a scalar!"

    # Discriminator loss should equal the average of real + fake
    expected_disc = (real_loss + fake_loss) * 0.5
    assert torch.isclose(disc_loss, expected_disc), \
        "Discriminator loss averaging mismatch!"

    # -----------------------------------------------------------------------
    # Optimiser factory — use tiny linear modules as stand-ins
    # -----------------------------------------------------------------------
    dummy_generator     = nn.Linear(10, 10)
    dummy_discriminator = nn.Linear(10, 10)

    gen_opt, disc_opt = create_optimizers(dummy_generator, dummy_discriminator)

    print("\n--- Optimisers ---")
    print("  Optimizer Created Successfully")
    print(f"  Generator     lr={gen_opt.param_groups[0]['lr']}  "
          f"betas={gen_opt.param_groups[0]['betas']}")
    print(f"  Discriminator lr={disc_opt.param_groups[0]['lr']}  "
          f"betas={disc_opt.param_groups[0]['betas']}")

    assert isinstance(gen_opt, Adam),  "Generator optimiser must be Adam!"
    assert isinstance(disc_opt, Adam), "Discriminator optimiser must be Adam!"
    assert gen_opt.param_groups[0]["lr"]    == ADAM_LR
    assert disc_opt.param_groups[0]["lr"]   == ADAM_LR
    assert gen_opt.param_groups[0]["betas"] == ADAM_BETAS

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print("\n[OK] All checks passed.")
    print("=" * 50 + "\n")
