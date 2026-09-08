"""Experiment configuration.

Every training run is described by one of these, loaded from YAML. Nothing that
affects a result lives in code, so a run can be reproduced from its config file
alone, and a Colab run and a local run differ only in the file they were handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Where the data is and how it is split."""

    archive: Path | None = None
    """Packaged archive to train from. When unset, splits are built from data/raw."""

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    min_samples_per_class: int = 20

    binary: bool = True
    """Collapse to healthy versus affected. This is the task the product performs."""

    balance_classes: bool = True
    """Weight the loss by inverse class frequency."""


class ModelConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3


class OptimConfig(BaseModel):
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    """Epochs with the backbone frozen, to settle the fresh head."""

    label_smoothing: float = 0.05
    grad_clip: float | None = 1.0
    amp: bool = True
    """Mixed precision. Roughly halves memory, which matters on a 4 GB card."""

    early_stopping_patience: int | None = 5


class TrainingConfig(BaseModel):
    """Full description of one experiment."""

    name: str = Field(description="Run name, used for the output directory.")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()

    field_prevalence: float = 0.02
    """Real-world prevalence used to project PPV. The datasets are ~80% affected,
    so metrics computed on them mean nothing until projected to this."""

    wandb_project: str | None = None
    """When set, metrics are logged to Weights & Biases."""

    @model_validator(mode="after")
    def check_fractions(self) -> TrainingConfig:
        total = self.data.val_fraction + self.data.test_fraction
        if not 0.0 < total < 1.0:
            raise ValueError(f"val + test fractions must be in (0, 1), got {total}")
        if self.optim.warmup_epochs >= self.optim.epochs:
            raise ValueError(
                f"warmup_epochs ({self.optim.warmup_epochs}) must be less than "
                f"epochs ({self.optim.epochs})"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load a config from a YAML file."""
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        """Write this config beside a run's outputs, so the run is reproducible."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
