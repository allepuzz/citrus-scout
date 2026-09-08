"""Command line interface."""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
