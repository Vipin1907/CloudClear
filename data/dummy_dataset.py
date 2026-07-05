"""
data/dummy_dataset.py
=====================
Dummy Dataset module for the CloudClear-NER project.

Generates synthetic paired (cloudy, cloud-free) satellite image patches
entirely in-memory using torch.rand(), allowing the full AI pipeline to be
developed and tested before real satellite data becomes available.

Author : CloudClear-NER Team
Created: 2026-07-05
"""

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DummyCloudDataset(Dataset):
    """
    Synthetic paired dataset for cloud-removal pipeline testing.

    Each sample consists of:

    * **cloudy**  – a (3, 256, 256) float32 tensor in [-1, 1] that simulates
                    a multispectral satellite patch partially obscured by
                    synthetic cloud noise.
    * **target**  – a (3, 256, 256) float32 tensor in [-1, 1] that represents
                    the corresponding cloud-free ground-truth patch.

    .. note::
        Values are in the **[-1, 1]** range to match the ``nn.Tanh`` output
        of ``UNetGenerator``. Normalise to [0, 1] (``x * 0.5 + 0.5``) only
        for display or metric computation.

    Everything is generated on-the-fly with ``torch.rand()``; no image files
    are read and no external datasets are required.

    Parameters
    ----------
    length : int, optional
        Number of samples in the dataset. Defaults to 100.
    image_size : int, optional
        Spatial resolution (height and width) of each patch. Defaults to 256.
    channels : int, optional
        Number of spectral channels per patch. Defaults to 3.
    seed : int or None, optional
        If provided, fixes the global torch seed so results are reproducible
        across runs. Defaults to ``None`` (non-deterministic).

    Examples
    --------
    >>> dataset = DummyCloudDataset(length=50)
    >>> cloudy, target = dataset[0]
    >>> cloudy.shape
    torch.Size([3, 256, 256])
    >>> target.shape
    torch.Size([3, 256, 256])
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        length: int = 100,
        image_size: int = 256,
        channels: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if length <= 0:
            raise ValueError(f"'length' must be a positive integer, got {length}.")
        if image_size <= 0:
            raise ValueError(f"'image_size' must be a positive integer, got {image_size}.")
        if channels <= 0:
            raise ValueError(f"'channels' must be a positive integer, got {channels}.")

        self.length: int = length
        self.image_size: int = image_size
        self.channels: int = channels

        # Optional reproducibility seed
        if seed is not None:
            torch.manual_seed(seed)

        # Pre-generate all samples once and cache them in CPU memory.
        # Shape: (length, channels, image_size, image_size)
        self._targets: torch.Tensor = self._generate_targets()
        self._cloudys: torch.Tensor = self._apply_cloud_noise(self._targets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_targets(self) -> torch.Tensor:
        """
        Generate cloud-free ground-truth patches.

        Values are mapped to the **[-1, 1]** range (consistent with
        ``nn.Tanh``) via ``torch.rand() * 2 - 1``.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape (length, C, H, W) with values in [-1, 1].
        """
        return torch.rand(
            self.length,
            self.channels,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        ) * 2 - 1

    def _apply_cloud_noise(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Simulate cloud contamination by overlaying synthetic noise on the
        clean patches and clamping the result to [-1, 1].

        Cloud effect is modelled as additive noise (also in [-1, 1]) scaled
        by a per-sample cloud-density factor drawn from Uniform(0.2, 0.6).
        The final clamp keeps all values inside the Tanh output range.

        Parameters
        ----------
        targets : torch.Tensor
            Clean patches of shape (length, C, H, W) with values in [-1, 1].

        Returns
        -------
        torch.Tensor
            Cloudy patches of shape (length, C, H, W) with values in [-1, 1].
        """
        # Per-sample cloud-density: shape (length, 1, 1, 1) for broadcasting
        density: torch.Tensor = torch.empty(
            self.length, 1, 1, 1, dtype=torch.float32
        ).uniform_(0.2, 0.6)

        # Noise also lives in [-1, 1] to stay consistent with the target range
        noise: torch.Tensor = torch.rand_like(targets) * 2 - 1

        cloudy: torch.Tensor = targets + density * noise
        return cloudy.clamp_(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single (cloudy, target) sample pair.

        Parameters
        ----------
        index : int
            Sample index in the range ``[0, len(dataset))``.

        Returns
        -------
        cloudy : torch.Tensor
            Cloudy input patch of shape (C, H, W), float32, values in [-1, 1].
        target : torch.Tensor
            Cloud-free target patch of shape (C, H, W), float32, values in [-1, 1].
        """
        return self._cloudys[index], self._targets[index]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"length={self.length}, "
            f"channels={self.channels}, "
            f"image_size={self.image_size})"
        )


# ---------------------------------------------------------------------------
# Quick-start test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal smoke-test that exercises the dataset and DataLoader.

    Expected output:
        [Dataset] DummyCloudDataset(length=20, channels=3, image_size=256)
        [Batch 1]
          Cloudy shape : torch.Size([4, 3, 256, 256])
          Target shape : torch.Size([4, 3, 256, 256])
          Cloudy range : [-1.0000, 1.0000]
          Target range : [-1.0000, 1.0000]
    """

    # ---- Create dataset ------------------------------------------------
    dataset = DummyCloudDataset(length=20)
    print(f"\n[Dataset] {dataset}")

    # ---- Wrap in DataLoader --------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # ---- Inspect the first batch ---------------------------------------
    cloudy_batch, target_batch = next(iter(loader))

    print("\n[Batch 1]")
    print(f"  Cloudy shape : {cloudy_batch.shape}")
    print(f"  Target shape : {target_batch.shape}")
    print(
        f"  Cloudy range : "
        f"[{cloudy_batch.min():.4f}, {cloudy_batch.max():.4f}]"
    )
    print(
        f"  Target range : "
        f"[{target_batch.min():.4f}, {target_batch.max():.4f}]"
    )

    # ---- Assertions (fail loud if anything is wrong) -------------------
    assert cloudy_batch.shape == torch.Size([4, 3, 256, 256]), \
        f"Unexpected cloudy shape: {cloudy_batch.shape}"
    assert target_batch.shape == torch.Size([4, 3, 256, 256]), \
        f"Unexpected target shape: {target_batch.shape}"
    assert cloudy_batch.min() >= -1.0 and cloudy_batch.max() <= 1.0, \
        "Cloudy values are out of [-1, 1]!"
    assert target_batch.min() >= -1.0 and target_batch.max() <= 1.0, \
        "Target values are out of [-1, 1]!"

    print("\n[OK] All assertions passed -- DummyCloudDataset is ready.\n")
