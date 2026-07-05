"""
train_dummy.py
==============
Dummy training pipeline for the CloudClear-NER project.

Verifies that the complete deep learning workflow (data loading, forward pass,
loss computation, and weight updates) works correctly end-to-end before real
satellite patches become available.

Pipeline:
    DummyCloudDataset  ->  UNetGenerator  ->  L1Loss  ->  Adam  ->  1 epoch

Author : CloudClear-NER Team
Created: 2026-07-05
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup – allows running from the project root without installing the
# package, so ``python train_dummy.py`` always resolves the local modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dummy_dataset import DummyCloudDataset      # noqa: E402
from models.unet_generator import UNetGenerator       # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_LENGTH: int = 100       # total synthetic samples
BATCH_SIZE: int = 4             # samples per mini-batch
LEARNING_RATE: float = 1e-4     # Adam learning rate
NUM_EPOCHS: int = 1             # single epoch for pipeline validation


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Returns
    -------
    torch.device
        ``cuda`` if a CUDA-capable GPU is available, otherwise ``cpu``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")
    return device


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader() -> DataLoader:
    """
    Construct a DataLoader backed by DummyCloudDataset.

    Returns
    -------
    DataLoader
        Ready-to-iterate loader with ``batch_size=BATCH_SIZE`` and
        ``shuffle=True``.
    """
    dataset = DummyCloudDataset(length=DATASET_LENGTH)
    loader: DataLoader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # keeps things simple; bump up for real data
        pin_memory=False,       # set True when using CUDA for speed
    )
    return loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> nn.Module:
    """
    Instantiate UNetGenerator and move it to *device*.

    Returns
    -------
    nn.Module
        Model ready for training.
    """
    model = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_features=64,
    )
    model.to(device)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Execute a single training epoch.

    For each batch:
        1. Move tensors to device.
        2. Forward pass through UNetGenerator.
        3. Compute L1 loss against the clean target.
        4. Back-propagate gradients.
        5. Update model weights.

    Parameters
    ----------
    model     : UNetGenerator instance in train mode.
    loader    : DataLoader yielding (cloudy, target) pairs.
    criterion : Loss function (L1Loss).
    optimizer : Weight update rule (Adam).
    device    : Compute device (cpu / cuda).
    """
    total_batches: int = len(loader)

    for batch_idx, (cloudy, target) in enumerate(loader, start=1):

        # ---- Move data to the active device --------------------------------
        cloudy: torch.Tensor = cloudy.to(device)   # (B, 3, 256, 256) in [-1, 1]
        target: torch.Tensor = target.to(device)   # (B, 3, 256, 256) in [-1, 1]

        # ---- Forward pass --------------------------------------------------
        prediction: torch.Tensor = model(cloudy)   # (B, 3, 256, 256), Tanh output

        # ---- Loss computation ----------------------------------------------
        loss: torch.Tensor = criterion(prediction, target)

        # ---- Backward pass + weight update ---------------------------------
        optimizer.zero_grad()   # clear stale gradients
        loss.backward()         # compute gradients
        optimizer.step()        # apply weight update

        # ---- Logging -------------------------------------------------------
        print(
            f"Epoch [{1}/{NUM_EPOCHS}]  "
            f"Batch [{batch_idx:>{len(str(total_batches))}}/{total_batches}]  "
            f"Loss: {loss.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full dummy training pipeline:

    1. Detect device.
    2. Build DataLoader.
    3. Build model, loss, and optimiser.
    4. Run one training epoch.
    5. Report success.
    """
    # -- Device --------------------------------------------------------------
    device: torch.device = get_device()

    # -- Data ----------------------------------------------------------------
    loader: DataLoader = build_dataloader()

    # -- Model ---------------------------------------------------------------
    model: nn.Module = build_model(device)

    # -- Loss & Optimiser ----------------------------------------------------
    criterion: nn.Module = nn.L1Loss()

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -- Training ------------------------------------------------------------
    print("-" * 60)
    run_one_epoch(model, loader, criterion, optimizer, device)
    print("-" * 60)

    # -- Done ----------------------------------------------------------------
    print("\nTraining pipeline executed successfully.\n")


if __name__ == "__main__":
    main()
