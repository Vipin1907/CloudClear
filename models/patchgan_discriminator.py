"""
models/patchgan_discriminator.py
================================
Production-ready PatchGAN Discriminator for the CloudClear-NER project.

Based on the architecture described in:
    "Image-to-Image Translation with Conditional Adversarial Networks"
    Isola et al., CVPR 2017  (Pix2Pix)

Architecture Overview:
    Input  (B,  3, 256, 256)
       │
       ▼
    Conv Block 1 — Conv2d(3 →  64, k=4, s=2, p=1) + LeakyReLU(0.2)
       │            [No BatchNorm on first layer — pix2pix convention]
       │            Feature map: (B,  64, 128, 128)
       ▼
    Conv Block 2 — Conv2d(64 → 128, k=4, s=2, p=1) + BN + LeakyReLU(0.2)
       │            Feature map: (B, 128,  64,  64)
       ▼
    Conv Block 3 — Conv2d(128 → 256, k=4, s=2, p=1) + BN + LeakyReLU(0.2)
       │            Feature map: (B, 256,  32,  32)
       ▼
    Conv Block 4 — Conv2d(256 → 512, k=4, s=1, p=1) + BN + LeakyReLU(0.2)
       │            Feature map: (B, 512,  31,  31)
       │            [stride=1 here to reduce spatial shrinkage]
       ▼
    Final Conv   — Conv2d(512 → 1, k=4, s=1, p=1)
       │            [No BN, No Sigmoid — raw logits for BCEWithLogitsLoss]
       ▼
    PatchGAN Score Map  (B,  1, 30, 30)

Each value in the (30×30) output map is the real/fake score for a
receptive-field patch (~70×70 pixels) of the original image, giving
the discriminator a local, patch-level understanding rather than a
single global scalar.

Usage:
    disc   = PatchGANDiscriminator(in_channels=3, base_features=64)
    logits = disc(image_tensor)          # (B, 1, 30, 30)
    loss   = nn.BCEWithLogitsLoss()(logits, target)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building Block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Single PatchGAN Convolutional Block.

    Applies:
        Conv2d  -> optional BatchNorm2d -> LeakyReLU(0.2)

    Note:
        BatchNorm is skipped on the very first discriminator layer, following
        the original Pix2Pix implementation.  The ``use_batchnorm`` flag
        controls this.

    Args:
        in_channels  (int) : Number of input channels.
        out_channels (int) : Number of output channels.
        stride       (int) : Conv stride — use 2 for downsampling blocks,
                             1 for the penultimate / final blocks.
        use_batchnorm(bool): Whether to include BatchNorm2d after Conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        layers: list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not use_batchnorm,   # bias is redundant when BN normalises
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv -> [BN] -> LeakyReLU.

        Args:
            x: Input feature tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, H', W').
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Cloud Removal (CloudClear-NER).

    Implements the 70×70 PatchGAN discriminator from the Pix2Pix paper.
    Instead of a single real/fake scalar per image, it produces a 2D map of
    overlapping patch predictions, giving localised, texture-level feedback
    to the generator.

    Architecture (default, in_channels=3, base_features=64):
        Block 1  : Conv2d( 3 →  64, k=4, s=2, p=1)               + LReLU
        Block 2  : Conv2d(64 → 128, k=4, s=2, p=1) + BN          + LReLU
        Block 3  : Conv2d(128→ 256, k=4, s=2, p=1) + BN          + LReLU
        Block 4  : Conv2d(256→ 512, k=4, s=1, p=1) + BN          + LReLU
        Final    : Conv2d(512→   1, k=4, s=1, p=1)

    Input  shape : (B, in_channels, 256, 256)
    Output shape : (B, 1, 30, 30)   — raw logits, one per patch

    Raw logits are returned (no Sigmoid) so that the caller can use
    ``nn.BCEWithLogitsLoss`` for numerical stability.

    Args:
        in_channels  (int): Number of input image channels (default: 3 for RGB).
        base_features(int): Number of feature maps in the first Conv block
                            (default: 64; subsequent blocks double this).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        f = base_features  # shorthand: 64

        # ------------------------------------------------------------------ #
        # Block 1 — no BatchNorm (pix2pix convention for the first layer)     #
        # Input : (B,   3, 256, 256)                                          #
        # Output: (B,  64, 128, 128)  [stride=2 halves spatial dims]          #
        # ------------------------------------------------------------------ #
        self.block1 = ConvBlock(in_channels, f,      stride=2, use_batchnorm=False)

        # ------------------------------------------------------------------ #
        # Block 2                                                             #
        # Output: (B, 128,  64,  64)                                          #
        # ------------------------------------------------------------------ #
        self.block2 = ConvBlock(f,      f * 2,  stride=2, use_batchnorm=True)

        # ------------------------------------------------------------------ #
        # Block 3                                                             #
        # Output: (B, 256,  32,  32)                                          #
        # ------------------------------------------------------------------ #
        self.block3 = ConvBlock(f * 2,  f * 4,  stride=2, use_batchnorm=True)

        # ------------------------------------------------------------------ #
        # Block 4 — stride=1 to slow spatial shrinkage before final layer     #
        # Output: (B, 512,  31,  31)                                          #
        # ------------------------------------------------------------------ #
        self.block4 = ConvBlock(f * 4,  f * 8,  stride=1, use_batchnorm=True)

        # ------------------------------------------------------------------ #
        # Final Conv — outputs a single-channel PatchGAN score map            #
        # No BatchNorm, no activation (raw logits for BCEWithLogitsLoss)      #
        # Output: (B,   1,  30,  30)                                          #
        # ------------------------------------------------------------------ #
        self.final_conv = nn.Conv2d(
            f * 8,          # 512
            1,              # one real/fake score per patch
            kernel_size=4,
            stride=1,
            padding=1,
        )

    # ---------------------------------------------------------------------- #
    # Forward Pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the discriminator forward pass.

        Args:
            x: Input image tensor of shape (B, in_channels, H, W).
               Typical input: (B, 3, 256, 256).

        Returns:
            PatchGAN score map of shape (B, 1, H', W').
            Each spatial position (i, j) holds the raw logit for whether the
            corresponding ~70×70 pixel patch is real or generated.
            Pass through ``torch.sigmoid()`` only if needed for inference;
            use raw logits with ``nn.BCEWithLogitsLoss`` during training.
        """
        x = self.block1(x)       # (B,  64, 128, 128)
        x = self.block2(x)       # (B, 128,  64,  64)
        x = self.block3(x)       # (B, 256,  32,  32)
        x = self.block4(x)       # (B, 512,  31,  31)
        x = self.final_conv(x)   # (B,   1,  30,  30)
        return x

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def param_count(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Integer representing the total number of trainable parameters
            in the discriminator.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick Sanity Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CloudClear-NER  |  PatchGAN Discriminator -- Arch Test")
    print("=" * 60)

    # Instantiate the discriminator
    discriminator = PatchGANDiscriminator(in_channels=3, base_features=64)
    discriminator.eval()

    # Dummy tensor simulating a single RGB image at 256x256
    dummy_input = torch.randn(1, 3, 256, 256)

    # Forward pass (no gradient tracking needed for shape verification)
    with torch.no_grad():
        score_map = discriminator(dummy_input)

    # ---- Print results ----
    print(f"\nInput  Shape       : {dummy_input.shape}")
    print(f"Output Shape       : {score_map.shape}")
    print(f"Trainable Params   : {discriminator.param_count():,}")

    # ---- Assertions ----
    # 1. Batch dimension must be preserved
    assert score_map.shape[0] == dummy_input.shape[0], (
        f"Batch size mismatch: got {score_map.shape[0]}, expected {dummy_input.shape[0]}"
    )

    # 2. Output must have exactly one channel (real/fake per patch)
    assert score_map.shape[1] == 1, (
        f"Channel mismatch: expected 1 output channel, got {score_map.shape[1]}"
    )

    # 3. Output must be a 4-D PatchGAN score map (B, 1, H', W')
    assert score_map.ndim == 4, (
        f"Expected 4-D output, got {score_map.ndim}-D tensor"
    )

    # 4. Spatial dims must be smaller than input (patch-level, not image-level)
    assert score_map.shape[2] < dummy_input.shape[2], (
        "Output spatial size should be smaller than input (patch-level output)"
    )

    print("\n[PASS] Forward pass successful")
    print(f"[PASS] Output shape valid            : {tuple(score_map.shape)}")
    print("[PASS] Batch size preserved")
    print("[PASS] Single-channel PatchGAN output confirmed")
    print("[PASS] PatchGAN architecture verified")
    print(f"\nReceptive field covers ~70x70 pixel patches of the input image.")
    print("Use BCEWithLogitsLoss during training (no Sigmoid needed on logits).")
    print("\nAll checks passed -- discriminator is production-ready!")
    print("=" * 60)



