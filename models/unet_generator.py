"""
models/unet_generator.py
========================
Production-ready U-Net Generator for the CloudClear-NER project.

Architecture Overview:
    Input  : (B, 3,   256, 256)
    Encoder: 5 DownBlocks  →  3 → 64 → 128 → 256 → 512 → 512
    Bottleneck              →  512 → 512
    Decoder: 5 UpBlocks    →  512 → 512 → 256 → 128 → 64 → 3
    Output : (B, 3,   256, 256)  with Tanh activation

Skip connections concatenate each encoder feature map with its
corresponding decoder feature map, preserving fine spatial detail
(edges, textures) critical for cloud-removal tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Encoder (Downsampling) Block.

    Applies:
        Conv2d(kernel=4, stride=2, padding=1)  -> spatial dims halved
        BatchNorm2d                             -> optional (skip on 1st block)
        LeakyReLU(0.2)                          -> activation

    Args:
        in_channels  : Number of input feature channels.
        out_channels : Number of output feature channels (doubled each block).
        use_batchnorm: If False, BatchNorm is omitted (e.g., first encoder block).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        layers: list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_batchnorm,  # bias is redundant when BN is used
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder (Upsampling) Block.

    Applies:
        ConvTranspose2d(kernel=4, stride=2, padding=1) -> spatial dims doubled
        BatchNorm2d
        ReLU

    Skip connections are handled *outside* this block via torch.cat() before
    calling forward(), so in_channels must account for the concatenated channels.

    Args:
        in_channels  : Channels coming in (includes skip-connection channels).
        out_channels : Number of output feature channels.
        use_dropout  : If True, adds Dropout(0.5) for regularisation (deep layers).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        layers: list = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if use_dropout:
            layers.append(nn.Dropout(p=0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net Generator for Cloud Removal (CloudClear-NER).

    Input  Shape : (B, in_channels,  H, W)   -- default (B, 3, 256, 256)
    Output Shape : (B, out_channels, H, W)   -- default (B, 3, 256, 256)

    Channel Progression:
        Encoder  : 3 -> 64 -> 128 -> 256 -> 512 -> 512
        Bottleneck: 512 -> 512
        Decoder  : (512+512) -> 512 -> (512+512) -> 256 -> (256+256) -> 128
                   -> (128+128) -> 64 -> (64+64) -> 3

    Note: Decoder in_channels double at each level because the skip-connection
    feature map (from the corresponding encoder block) is concatenated before
    every UpBlock, following the original U-Net design.

    Args:
        in_channels  : Input image channels  (default: 3 for RGB).
        out_channels : Output image channels (default: 3 for RGB).
        base_features: Base number of feature channels (default: 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        f = base_features  # shorthand: 64

        # ------------------------------------------------------------------ #
        # Encoder -- 5 Downsampling blocks                                    #
        # Each block halves H x W and (mostly) doubles channels.              #
        # ------------------------------------------------------------------ #
        # Block 1: No BatchNorm on the very first encoder layer (pix2pix convention)
        self.enc1 = DownBlock(in_channels, f,      use_batchnorm=False)  # ->  64, 128x128
        self.enc2 = DownBlock(f,           f * 2,  use_batchnorm=True)   # -> 128,  64x64
        self.enc3 = DownBlock(f * 2,       f * 4,  use_batchnorm=True)   # -> 256,  32x32
        self.enc4 = DownBlock(f * 4,       f * 8,  use_batchnorm=True)   # -> 512,  16x16
        self.enc5 = DownBlock(f * 8,       f * 8,  use_batchnorm=True)   # -> 512,   8x8

        # ------------------------------------------------------------------ #
        # Bottleneck -- deepest representation, no skip connection here       #
        # ------------------------------------------------------------------ #
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),  # -> 512, 4x4
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # ------------------------------------------------------------------ #
        # Decoder -- 5 Upsampling blocks with skip connections                #
        # in_channels = decoder_channels + skip_channels (torch.cat)          #
        # ------------------------------------------------------------------ #
        # up1 receives bottleneck output only (no skip at bottleneck level)
        self.up1 = UpBlock(f * 8,          f * 8, use_dropout=True)   # 512       -> 512,  8x8
        # From up1 onward each UpBlock input = its own output + enc skip
        self.up2 = UpBlock(f * 8 + f * 8,  f * 8, use_dropout=True)   # 512+512   -> 512, 16x16
        self.up3 = UpBlock(f * 8 + f * 8,  f * 4, use_dropout=False)  # 512+512   -> 256, 32x32
        self.up4 = UpBlock(f * 4 + f * 4,  f * 2, use_dropout=False)  # 256+256   -> 128, 64x64
        self.up5 = UpBlock(f * 2 + f * 2,  f,     use_dropout=False)  # 128+128   ->  64,128x128

        # ------------------------------------------------------------------ #
        # Output layer                                                        #
        # in_channels = up5 output + enc1 skip                                #
        # ------------------------------------------------------------------ #
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                f + f,          # 64 + 64 = 128
                out_channels,   # 3
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),          # outputs in [-1, 1], matching normalised images
        )

    # ---------------------------------------------------------------------- #
    # Forward Pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the U-Net.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Reconstructed tensor of shape (B, out_channels, H, W).
        """
        # ---- Encoder ----
        e1 = self.enc1(x)           # (B,  64, 128, 128)
        e2 = self.enc2(e1)          # (B, 128,  64,  64)
        e3 = self.enc3(e2)          # (B, 256,  32,  32)
        e4 = self.enc4(e3)          # (B, 512,  16,  16)
        e5 = self.enc5(e4)          # (B, 512,   8,   8)

        # ---- Bottleneck ----
        bn = self.bottleneck(e5)    # (B, 512,   4,   4)

        # ---- Decoder with skip connections ----
        # up1: no skip (bottleneck has no paired encoder block)
        d1 = self.up1(bn)                               # (B, 512,   8,   8)

        # Concatenate corresponding encoder features along channel dim
        d2 = self.up2(torch.cat((d1, e5), dim=1))      # (B, 512,  16,  16)
        d3 = self.up3(torch.cat((d2, e4), dim=1))      # (B, 256,  32,  32)
        d4 = self.up4(torch.cat((d3, e3), dim=1))      # (B, 128,  64,  64)
        d5 = self.up5(torch.cat((d4, e2), dim=1))      # (B,  64, 128, 128)

        # ---- Output ----
        out = self.output_layer(torch.cat((d5, e1), dim=1))  # (B, 3, 256, 256)
        return out

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def param_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick Sanity Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  CloudClear-NER  |  U-Net Generator -- Architecture Test")
    print("=" * 55)

    # Instantiate model (CPU is fine for shape verification)
    model = UNetGenerator(in_channels=3, out_channels=3, base_features=64)
    model.eval()

    # Create a random dummy tensor simulating a batch of 1 RGB image
    dummy_input = torch.randn(1, 3, 256, 256)

    # Forward pass (no gradient tracking needed for shape check)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput  Shape : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")
    print(f"Trainable Params : {model.param_count():,}")

    # Assertions
    assert output.shape == dummy_input.shape, (
        f"Shape mismatch! Expected {dummy_input.shape}, got {output.shape}"
    )
    assert output.min() >= -1.0 and output.max() <= 1.0, (
        "Tanh output out of [-1, 1] range!"
    )

    print("\n[PASS] No tensor size mismatch")
    print("[PASS] Skip connections align correctly")
    print("[PASS] Forward pass runs successfully")
    print("[PASS] Output values in Tanh range [-1, 1]")
    print("\nAll checks passed -- architecture is valid!")
    print("=" * 55)
