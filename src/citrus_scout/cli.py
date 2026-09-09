"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from citrus_scout.data.sources import CATALOGUE
from citrus_scout.data.taxonomy import relevance_of

app = typer.Typer(
    help="Pest and disease detection in citrus from UAV imagery.", no_args_is_help=True
)
data_app = typer.Typer(help="Dataset download and inspection.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()


@data_app.command("download")
def download_command(
    force: bool = typer.Option(False, "--force", help="Re-download even if data is present."),
    include_noncommercial: bool = typer.Option(
        False,
        "--include-noncommercial",
        help="Also fetch datasets whose licence does not clearly allow commercial use.",
    ),
) -> None:
    """Download the public leaf datasets into data/raw/."""
    from citrus_scout.data.download import KaggleAuthError, download_all

    try:
        results = download_all(force=force, commercial_only=not include_noncommercial)
    except KaggleAuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    for key, path in results.items():
        console.print(f"[green]ok[/green] {key} -> {path}")


@data_app.command("sources")
def sources_command() -> None:
    """List the dataset catalogue and its licences."""
    table = Table(title="Dataset catalogue")
    table.add_column("key")
    table.add_column("licence")
    table.add_column("size", justify="right")
    table.add_column("commercial", justify="center")

    for source in CATALOGUE:
        table.add_row(
            source.key,
            source.licence,
            f"{source.approx_size_mb} MB",
            "yes" if source.is_commercially_usable else "no",
        )
    console.print(table)


@data_app.command("summary")
def summary_command(
    include_augmented: bool = typer.Option(
        False,
        "--include-augmented",
        help="Include pre-augmented copies. Off by default: they leak across splits.",
    ),
    seed: int = typer.Option(42, help="Split seed."),
    min_samples: int = typer.Option(20, help="Drop classes with fewer samples than this."),
) -> None:
    """Build the splits and print what the combined dataset contains."""
    from citrus_scout.data.build import binary_labels, build_splits

    split, summary = build_splits(
        include_augmented=include_augmented,
        seed=seed,
        min_samples_per_class=min_samples,
    )
    console.print(summary.describe())
    console.print()

    for name, group in (("train", split.train), ("val", split.val), ("test", split.test)):
        labels = binary_labels(group)
        prevalence = sum(labels) / len(labels)
        console.print(
            f"{name:6s} n={len(labels):5d}  affected={sum(labels):5d}  prevalence={prevalence:.1%}"
        )

    console.print(
        "\n[yellow]Note:[/yellow] prevalence here is a property of how these datasets "
        "were collected, not of a real grove, where 2-5% is typical. Metrics must be "
        "projected to field prevalence before they mean anything."
    )


@data_app.command("relevance")
def relevance_command() -> None:
    """Show which classes correspond to pests and diseases present in Murcia."""
    from citrus_scout.data.build import build_splits
    from citrus_scout.data.taxonomy import RELEVANCE_NOTES

    _, summary = build_splits()

    table = Table(title="Class relevance for the Region of Murcia")
    table.add_column("class")
    table.add_column("images", justify="right")
    table.add_column("relevance")
    table.add_column("note")

    for label, count in sorted(summary.per_label.items(), key=lambda x: -x[1]):
        table.add_row(
            label,
            str(count),
            relevance_of(label).value,
            RELEVANCE_NOTES.get(label, ""),
        )
    console.print(table)


@data_app.command("package")
def package_command(
    output: Path = typer.Option(
        Path("data/processed/leaf_dataset.zip"),
        "--output",
        "-o",
        help="Where to write the archive.",
    ),
    max_side: int = typer.Option(512, help="Downscale so the longest side is at most this."),
    quality: int = typer.Option(90, help="JPEG quality for the export."),
    seed: int = typer.Option(42, help="Split seed, baked into the archive."),
) -> None:
    """Package the split images into one archive for upload to Colab.

    Uploading the raw download to Drive is impractical: thousands of files where
    per-file latency dominates. This writes a single downscaled archive instead.
    """
    from citrus_scout.data.package import package_for_colab

    with console.status("packaging images..."):
        stats = package_for_colab(output, max_side=max_side, quality=quality, seed=seed)
    console.print(f"[green]ok[/green] {stats.describe()}")
    console.print(f"upload this file to Drive: {stats.archive}")


@app.command("train")
def train_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML config."),
    archive: Path | None = typer.Option(
        None, "--archive", help="Train from a packaged archive instead of data/raw."
    ),
    epochs: int | None = typer.Option(None, help="Override the epoch count."),
    batch_size: int | None = typer.Option(None, help="Override the batch size."),
    wandb_project: str | None = typer.Option(None, help="Log metrics to this W&B project."),
) -> None:
    """Train a model from a config file."""
    from citrus_scout.training.config import TrainingConfig
    from citrus_scout.training.train import run_training

    settings = TrainingConfig.from_yaml(config)

    # Overrides exist so a Colab cell can vary batch size for a different GPU
    # without editing the versioned config.
    if archive is not None:
        settings.data.archive = archive
    if epochs is not None:
        settings.optim.epochs = epochs
    if batch_size is not None:
        settings.data.batch_size = batch_size
    if wandb_project is not None:
        settings.wandb_project = wandb_project

    run_training(settings)


@app.command("evaluate")
def evaluate_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to a saved model."),
    archive: Path | None = typer.Option(None, "--archive", help="Packaged archive to use."),
    split: str = typer.Option("test", help="Which split to evaluate."),
    calibrate_on: str | None = typer.Option(
        "val",
        "--calibrate-on",
        help="Split to fit the temperature on. Must differ from --split. Pass 'none' to skip.",
    ),
) -> None:
    """Evaluate a checkpoint and report metrics at field prevalence."""
    from citrus_scout.evaluation.report import evaluate_checkpoint

    fit_split = None if calibrate_on in (None, "none", "") else calibrate_on
    if fit_split == split:
        raise typer.BadParameter(
            f"--calibrate-on ({fit_split}) must differ from --split ({split}): "
            "fitting and measuring on the same data reports a calibration that "
            "does not hold anywhere else."
        )

    evaluate_checkpoint(
        checkpoint,
        archive=archive,
        split=split,
        calibrate_on=fit_split,
        console=console,
    )


@app.command("uncertainty")
def uncertainty_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to a saved model."),
    archive: Path | None = typer.Option(None, "--archive", help="Packaged archive to use."),
    split: str = typer.Option("test", help="Which split to evaluate."),
    passes: int = typer.Option(20, help="Stochastic forward passes. Cost scales linearly."),
    target_specificity: float = typer.Option(
        0.99, help="Specificity defining the alert threshold before triage."
    ),
) -> None:
    """Quantify uncertainty with MC Dropout and triage predictions by confidence."""
    from citrus_scout.evaluation.report import report_uncertainty

    report_uncertainty(
        checkpoint,
        archive=archive,
        split=split,
        passes=passes,
        target_specificity=target_specificity,
        console=console,
    )


@app.command("per-class")
def per_class_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to a saved model."),
    archive: Path | None = typer.Option(None, "--archive", help="Packaged archive to use."),
    split: str = typer.Option("test", help="Which split to evaluate."),
    show_matrix: bool = typer.Option(True, help="Print the confusion matrix."),
) -> None:
    """Report per-class recall and what the model confuses, tagged by local relevance."""
    from citrus_scout.evaluation.report import report_per_class

    report_per_class(
        checkpoint,
        archive=archive,
        split=split,
        show_matrix=show_matrix,
        console=console,
    )


if __name__ == "__main__":
    app()
