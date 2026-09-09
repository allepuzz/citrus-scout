"""Run one training experiment end to end."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from citrus_scout.data.archive_dataset import load_packaged_splits
from citrus_scout.data.build import binary_labels, build_splits
from citrus_scout.data.leaf_dataset import LeafDataset
from citrus_scout.data.transforms import eval_transform, train_transform
from citrus_scout.models.classifier import (
    build_model,
    count_parameters,
    freeze_backbone,
    select_device,
    unfreeze_all,
)
from citrus_scout.training.config import TrainingConfig
from citrus_scout.training.loop import (
    EpochResult,
    class_weights,
    evaluate,
    format_duration,
    now,
    save_checkpoint,
    should_stop,
    train_one_epoch,
    write_history,
)
from citrus_scout.utils.paths import RUNS_DIR


def seed_everything(seed: int) -> None:
    """Fix every random source we control.

    cudnn.benchmark stays off: it picks algorithms based on timing, which makes runs
    non-deterministic. The throughput cost is small next to being unable to
    reproduce a result.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BinaryLabelWrapper(torch.utils.data.Dataset):
    """Present a multi-class dataset as healthy versus affected.

    Wrapping rather than relabelling keeps the original class of every sample
    available for error analysis, which is where the interesting failures show up:
    knowing the model confuses sooty mould with healthy is actionable, knowing it
    got a "1" wrong is not.
    """

    def __init__(self, dataset: LeafDataset, binary: list[int]) -> None:
        if len(dataset) != len(binary):
            raise ValueError(f"dataset has {len(dataset)} samples but {len(binary)} binary labels")
        self.dataset = dataset
        self.binary = binary
        self.classes = ["healthy", "affected"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        return image, self.binary[index]


def build_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    """Assemble train, validation and test loaders.

    Returns the loaders, the class names, and the training labels needed for class
    weighting.
    """
    if config.data.archive is not None:
        splits = load_packaged_splits(
            config.data.archive,
            train_tf=train_transform(config.data.image_size),
            eval_tf=eval_transform(config.data.image_size),
            binary=config.data.binary,
        )
        train_ds, val_ds, test_ds, classes = splits
        train_labels = [int(label) for _, label in train_ds.label_pairs()]
    else:
        split, _ = build_splits(
            seed=config.data.seed,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            min_samples_per_class=config.data.min_samples_per_class,
        )
        all_classes = sorted({s.label for s in split.train + split.val + split.test})

        raw_train = LeafDataset(
            split.train, classes=all_classes, transform=train_transform(config.data.image_size)
        )
        raw_val = LeafDataset(
            split.val, classes=all_classes, transform=eval_transform(config.data.image_size)
        )
        raw_test = LeafDataset(
            split.test, classes=all_classes, transform=eval_transform(config.data.image_size)
        )

        if config.data.binary:
            train_ds = BinaryLabelWrapper(raw_train, binary_labels(split.train))
            val_ds = BinaryLabelWrapper(raw_val, binary_labels(split.val))
            test_ds = BinaryLabelWrapper(raw_test, binary_labels(split.test))
            classes = ["healthy", "affected"]
            train_labels = binary_labels(split.train)
        else:
            train_ds, val_ds, test_ds = raw_train, raw_val, raw_test
            classes = all_classes
            train_labels = [raw_train.class_to_index[s.label] for s in split.train]

    workers = config.data.resolve_num_workers()
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, classes, train_labels


def run_training(config: TrainingConfig, *, output_dir: Path | None = None) -> Path:
    """Train a model according to `config`, returning its output directory."""
    seed_everything(config.data.seed)
    device = select_device(config.device)

    run_dir = output_dir or (RUNS_DIR / config.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / "config.yaml")

    train_loader, val_loader, test_loader, classes, train_labels = build_loaders(config)

    model = build_model(
        backbone=config.model.backbone,
        num_classes=len(classes),
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    trainable, total = count_parameters(model)
    print(f"device: {device}")
    print(f"classes: {classes}")
    print(f"model: {config.model.backbone}, {total / 1e6:.1f}M parameters")
    print(
        f"data: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    weights = None
    if config.data.balance_classes:
        weights = class_weights(train_labels, len(classes)).to(device)
        print(f"class weights: {[round(w, 3) for w in weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.optim.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.optim.epochs - config.optim.warmup_epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda") if config.optim.amp and device.type == "cuda" else None

    run = None
    if config.wandb_project:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            name=config.name,
            config=config.model_dump(mode="json"),
        )

    history: list[EpochResult] = []
    best_pr_auc = -1.0
    epochs_without_improvement = 0

    if config.optim.warmup_epochs > 0:
        freeze_backbone(model)
        print(f"warmup: backbone frozen for {config.optim.warmup_epochs} epoch(s)")

    for epoch in range(1, config.optim.epochs + 1):
        if epoch == config.optim.warmup_epochs + 1:
            unfreeze_all(model)
            trainable, total = count_parameters(model)
            print(f"backbone unfrozen: {trainable / 1e6:.1f}M trainable parameters")

        started = now()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            grad_clip=config.optim.grad_clip,
            description=f"epoch {epoch}/{config.optim.epochs}",
        )

        val_loss, report, _, _ = evaluate(
            model, val_loader, criterion, device, field_prevalence=config.field_prevalence
        )

        if epoch > config.optim.warmup_epochs:
            scheduler.step()

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_pr_auc=report["pr_auc"],
            val_f1=report["f1_at_0.5"],
            seconds=now() - started,
            extra={"lr": optimizer.param_groups[0]["lr"]},
        )
        history.append(result)
        print(result.describe())

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/pr_auc": report["pr_auc"],
                    "val/f1": report["f1_at_0.5"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        # Select on PR-AUC: validation loss can fall while minority-class ranking
        # gets worse, and ranking is what the operating threshold depends on.
        pr_auc = report["pr_auc"]

        # Ask before updating the best: `should_stop` judges this epoch against
        # the previous high-water mark.
        stop, epochs_without_improvement, reason = should_stop(
            pr_auc=pr_auc,
            best_pr_auc=best_pr_auc,
            epochs_without_improvement=epochs_without_improvement,
            patience=config.optim.early_stopping_patience,
            min_delta=config.optim.early_stopping_min_delta,
        )

        # The checkpoint follows any gain at all, with no floor applied.
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            save_checkpoint(run_dir / "best.pt", model, config, classes, epoch, report)

        if stop:
            print(f"stopping at epoch {epoch}: {reason}")
            break

    write_history(run_dir / "history.json", history)

    total_seconds = sum(r.seconds for r in history)
    print(f"\ntraining finished in {format_duration(total_seconds)}")
    print(f"best validation PR-AUC: {best_pr_auc:.4f}")
    print(f"outputs: {run_dir}")

    if run is not None:
        run.finish()

    return run_dir
