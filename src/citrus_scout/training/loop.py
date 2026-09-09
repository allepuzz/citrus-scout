"""Training loop.

Two details here are not boilerplate and matter for this problem:

**Class weighting.** The datasets run 80% affected. Left alone, the model learns
that predicting "affected" is usually right, which is the exact opposite of the
field situation it will be deployed into. Weighting the loss by inverse frequency
stops the majority class from dominating the gradient.

**Model selection on PR-AUC, not loss.** Validation loss can improve while the
model gets worse at the thing we care about, because loss rewards confident correct
predictions on the easy majority. PR-AUC tracks ranking quality on the minority
class, which is what the operating-point threshold is later chosen from.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from citrus_scout.evaluation.metrics import classification_report
from citrus_scout.training.config import TrainingConfig


@dataclass
class EpochResult:
    """Metrics from one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_pr_auc: float
    val_f1: float
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"epoch {self.epoch:3d} | train {self.train_loss:.4f} | "
            f"val {self.val_loss:.4f} | PR-AUC {self.val_pr_auc:.4f} | "
            f"F1 {self.val_f1:.4f} | {self.seconds:.0f}s"
        )


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Normalising keeps the loss on a comparable scale to the unweighted case, so the
    learning rate does not need retuning when weighting is toggled.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor(
        [total / (num_classes * max(counts.get(c, 0), 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    field_prevalence: float = 0.02,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Run the model over a loader, returning loss, metrics, scores and labels."""
    model.eval()

    losses: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        # Probability of the positive (affected) class.
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        all_scores.append(probabilities.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels_array = np.concatenate(all_labels)

    report = classification_report(labels_array, scores, field_prevalence=field_prevalence)
    return float(np.mean(losses)), report, scores, labels_array


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = None,
    description: str = "train",
) -> float:
    """One pass over the training set, returning mean loss."""
    model.train()
    losses: list[float] = []

    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                # Unscale before clipping, or the threshold applies to scaled gradients.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    classes: list[str],
    epoch: int,
    metrics: dict,
) -> None:
    """Persist weights alongside everything needed to interpret them.

    The class list is stored with the weights: a checkpoint whose class order is
    unknown produces confidently wrong predictions rather than an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.model_dump(mode="json"),
            "classes": classes,
            "epoch": epoch,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, int | float)},
        },
        path,
    )


def write_history(path: Path, history: list[EpochResult]) -> None:
    """Write per-epoch metrics as JSON for later plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "epoch": r.epoch,
            "train_loss": r.train_loss,
            "val_loss": r.val_loss,
            "val_pr_auc": r.val_pr_auc,
            "val_f1": r.val_f1,
            "seconds": r.seconds,
            **r.extra,
        }
        for r in history
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def now() -> float:
    return time.perf_counter()
