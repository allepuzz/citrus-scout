"""Evaluate a trained checkpoint and report what it would do in the field.

The headline number here is not accuracy or even PR-AUC: it is the positive
predictive value projected to real grove prevalence. A model can look excellent on
this test set, which runs 80% affected, and still raise six false alerts for every
true one at the 2% prevalence a technician actually faces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader

from citrus_scout.data.archive_dataset import load_packaged_splits
from citrus_scout.data.build import binary_labels, build_splits
from citrus_scout.data.leaf_dataset import LeafDataset
from citrus_scout.data.transforms import eval_transform
from citrus_scout.evaluation.calibration import (
    apply_temperature,
    calibrate,
    expected_calibration_error,
)
from citrus_scout.evaluation.metrics import classification_report
from citrus_scout.evaluation.uncertainty import (
    mc_dropout_predict,
    uncertainty_separates_errors,
)
from citrus_scout.models.classifier import build_model, select_device
from citrus_scout.training.config import TrainingConfig
from citrus_scout.training.loop import collect_logits, evaluate


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, TrainingConfig, list[str]]:
    """Restore a model together with the config and class list it was trained under."""
    payload = torch.load(path, map_location=device, weights_only=False)

    config = TrainingConfig.model_validate(payload["config"])
    classes = payload["classes"]

    model = build_model(
        backbone=config.model.backbone,
        num_classes=len(classes),
        pretrained=False,
        dropout=config.model.dropout,
    )
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), config, classes


def build_eval_loader(
    config: TrainingConfig,
    split_name: str,
    *,
    archive: Path | None = None,
) -> DataLoader:
    """Build a loader for one split, using the same preprocessing as training."""
    transform = eval_transform(config.data.image_size)
    source = archive or config.data.archive

    if source is not None:
        train_ds, val_ds, test_ds, _ = load_packaged_splits(
            source, train_tf=transform, eval_tf=transform, binary=config.data.binary
        )
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split_name]
    else:
        split, _ = build_splits(
            seed=config.data.seed,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            min_samples_per_class=config.data.min_samples_per_class,
        )
        samples = {"train": split.train, "val": split.val, "test": split.test}[split_name]
        all_classes = sorted({s.label for s in split.train + split.val + split.test})
        raw = LeafDataset(samples, classes=all_classes, transform=transform)

        if config.data.binary:
            from citrus_scout.training.train import BinaryLabelWrapper

            dataset = BinaryLabelWrapper(raw, binary_labels(samples))
        else:
            dataset = raw

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.resolve_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    archive: Path | None = None,
    split: str = "test",
    calibrate_on: str | None = "val",
    console: Console | None = None,
) -> dict:
    """Evaluate a checkpoint and print a field-relevant report.

    When `calibrate_on` names a split, a temperature is fitted there and the
    operating points are recomputed from the calibrated probabilities. The fit
    split must differ from the evaluated one, or the reported calibration is
    measured on the data that produced it.
    """
    console = console or Console()
    device = select_device("auto")

    model, config, classes = load_checkpoint(checkpoint, device)
    loader = build_eval_loader(config, split, archive=archive)

    criterion = nn.CrossEntropyLoss()
    loss, report, _scores, labels = evaluate(
        model, loader, criterion, device, field_prevalence=config.field_prevalence
    )

    console.print(f"\ncheckpoint: {checkpoint}")
    console.print(f"split: {split}, n={len(labels)}, classes={classes}")
    console.print(f"loss: {loss:.4f}")

    summary = Table(title="Ranking quality")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("PR-AUC", f"{report['pr_auc']:.4f}")
    summary.add_row("ROC-AUC", f"{report['roc_auc']:.4f}")
    summary.add_row("F1 at 0.5", f"{report['f1_at_0.5']:.4f}")
    summary.add_row("test prevalence", f"{report['test_prevalence']:.1%}")
    console.print(summary)

    operating = Table(
        title=f"Operating points, PPV projected to {report['field_prevalence']:.1%} field prevalence"
    )
    operating.add_column("threshold", justify="right")
    operating.add_column("sensitivity", justify="right")
    operating.add_column("specificity", justify="right")
    operating.add_column("PPV", justify="right")
    operating.add_column("alerts per true case", justify="right")

    for point in report["operating_points"]:
        alerts = f"{1 / point.ppv:.1f}" if point.ppv > 0 else "inf"
        operating.add_row(
            f"{point.threshold:.3f}",
            f"{point.sensitivity:.1%}",
            f"{point.specificity:.1%}",
            f"{point.ppv:.1%}",
            alerts,
        )
    console.print(operating)

    if calibrate_on is not None and calibrate_on != split:
        report["calibration"] = _report_calibration(
            model,
            config,
            device,
            evaluated_split=split,
            fit_split=calibrate_on,
            archive=archive,
            console=console,
        )

    console.print(
        "\n[yellow]Read the PPV column, not PR-AUC.[/yellow] This test set is "
        f"{report['test_prevalence']:.0%} affected; a real grove runs "
        f"{report['field_prevalence']:.0%}. The last column is how many trees a "
        "technician would visit per genuine case found."
    )

    return report


def _report_calibration(
    model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    *,
    evaluated_split: str,
    fit_split: str,
    archive: Path | None,
    console: Console,
) -> dict:
    """Fit a temperature on `fit_split` and report its effect on `evaluated_split`.

    The temperature is fitted on one split and measured on another, which is the
    only order that gives an honest number. Ranking metrics are not recomputed:
    dividing logits by a positive scalar is monotonic, so PR-AUC and ROC-AUC cannot
    move. Thresholds can, which is the whole point.
    """
    fit_loader = build_eval_loader(config, fit_split, archive=archive)
    eval_loader = build_eval_loader(config, evaluated_split, archive=archive)

    fit_logits, fit_labels = collect_logits(model, fit_loader, device)
    eval_logits, eval_labels = collect_logits(model, eval_loader, device)

    temperature, fit_report = calibrate(fit_logits, fit_labels)

    raw = apply_temperature(eval_logits, 1.0)
    scaled = apply_temperature(eval_logits, temperature)

    ece_raw = expected_calibration_error(eval_labels, raw)
    ece_scaled = expected_calibration_error(eval_labels, scaled)

    table = Table(title=f"Calibration (T fitted on {fit_split}, measured on {evaluated_split})")
    table.add_column("metric")
    table.add_column("raw", justify="right")
    table.add_column("calibrated", justify="right")
    table.add_row("ECE", f"{ece_raw:.4f}", f"{ece_scaled:.4f}")
    table.add_row("temperature", "1.000", f"{temperature:.3f}")
    console.print(table)

    calibrated_report = classification_report(
        eval_labels, scaled, field_prevalence=config.field_prevalence
    )

    operating = Table(title="Operating points after calibration")
    operating.add_column("threshold", justify="right")
    operating.add_column("sensitivity", justify="right")
    operating.add_column("specificity", justify="right")
    operating.add_column("PPV", justify="right")
    for point in calibrated_report["operating_points"]:
        operating.add_row(
            f"{point.threshold:.3f}",
            f"{point.sensitivity:.1%}",
            f"{point.specificity:.1%}",
            f"{point.ppv:.1%}",
        )
    console.print(operating)

    if temperature > 1.05:
        console.print(
            f"[yellow]T={temperature:.2f} > 1: the model was overconfident.[/yellow] "
            "Raw probabilities sat closer to 0 and 1 than the outcomes justified."
        )
    elif temperature < 0.95:
        console.print(f"[yellow]T={temperature:.2f} < 1: the model was underconfident.[/yellow]")

    return {
        "temperature": temperature,
        "fit_split": fit_split,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_scaled,
        "fit_summary": str(fit_report),
        "operating_points": calibrated_report["operating_points"],
    }


def report_uncertainty(
    checkpoint: Path,
    *,
    archive: Path | None = None,
    split: str = "test",
    passes: int = 20,
    target_specificity: float = 0.99,
    console: Console | None = None,
) -> dict:
    """Run MC Dropout and report what the uncertainty would do to a technician's day.

    The table that matters is the triage split. A threshold alone gives one number,
    the alert count; uncertainty splits those alerts into the ones worth walking to
    and the ones the model is quietly unsure about.
    """
    console = console or Console()
    device = select_device("auto")

    model, config, classes = load_checkpoint(checkpoint, device)
    loader = build_eval_loader(config, split, archive=archive)

    console.print(f"\ncheckpoint: {checkpoint}")
    console.print(f"split: {split}, classes={classes}")
    console.print(f"running {passes} stochastic passes...")

    result = mc_dropout_predict(model, loader, device, passes=passes)

    summary = Table(title=f"Uncertainty over {passes} MC Dropout passes")
    summary.add_column("statistic")
    summary.add_column("mean", justify="right")
    summary.add_column("max", justify="right")
    summary.add_row(
        "std across passes",
        f"{result.std_probability.mean():.4f}",
        f"{result.std_probability.max():.4f}",
    )
    summary.add_row(
        "predictive entropy",
        f"{result.predictive_entropy.mean():.4f}",
        f"{result.predictive_entropy.max():.4f}",
    )
    summary.add_row(
        "epistemic (MI)",
        f"{result.mutual_information.mean():.4f}",
        f"{result.mutual_information.max():.4f}",
    )
    console.print(summary)

    separation = uncertainty_separates_errors(result)
    if separation["separates"] is None:
        console.print(
            f"[yellow]No errors on this split ({separation['n_errors']} wrong), so "
            "whether uncertainty tracks mistakes cannot be measured here.[/yellow] "
            "That is a property of the test set being easy, not evidence the signal works."
        )
    elif separation["separates"]:
        console.print(
            f"[green]Uncertainty tracks errors:[/green] mean epistemic uncertainty is "
            f"{separation['ratio']:.1f}x higher on the {separation['n_errors']} wrong "
            "predictions than on the right ones. Ranking the queue by it is worth the cost."
        )
    else:
        console.print(
            "[red]Uncertainty does not track errors on this split.[/red] Wrong predictions "
            "are no more uncertain than right ones, so the extra passes buy nothing and "
            "the ordering carries no information."
        )

    # Triage at the project's preferred operating point: high specificity, because
    # the cost of a false alert is a wasted walk.
    affected_threshold = (
        float(np.quantile(result.mean_probability[result.labels == 0], target_specificity))
        if (result.labels == 0).any()
        else 0.5
    )

    masks = result.triage(
        affected_threshold=affected_threshold,
        healthy_threshold=min(1.0 - affected_threshold, affected_threshold),
    )

    triage = Table(
        title=f"Triage at {target_specificity:.0%} specificity (threshold {affected_threshold:.3f})"
    )
    triage.add_column("bucket")
    triage.add_column("trees", justify="right")
    triage.add_column("share", justify="right")
    triage.add_column("what happens")

    total = len(result)
    for name, description in (
        ("act", "flagged, inspection pass"),
        ("inspect", "queued for a human look"),
        ("ignore", "skipped"),
    ):
        count = int(masks[name].sum())
        triage.add_row(name, str(count), f"{count / total:.1%}", description)
    console.print(triage)

    return {
        "passes": passes,
        "mean_std": float(result.std_probability.mean()),
        "mean_epistemic": float(result.mutual_information.mean()),
        "separation": separation,
        "triage_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "threshold": affected_threshold,
    }
