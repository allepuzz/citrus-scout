"""Assemble the combined leaf dataset from all downloaded sources."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from citrus_scout.data.leaf_dataset import LeafSample, discover_samples
from citrus_scout.data.sources import CATALOGUE
from citrus_scout.data.splits import SplitResult, stratified_split, verify_no_leakage
from citrus_scout.data.taxonomy import LocalRelevance, relevance_of
from citrus_scout.utils.paths import RAW_DATA_DIR


@dataclass(frozen=True)
class DatasetSummary:
    """What was assembled, for logging and for the model card."""

    total_images: int
    per_source: dict[str, int]
    per_label: dict[str, int]
    per_relevance: dict[str, int]
    duplicates_removed: int
    split_sizes: dict[str, int]

    def describe(self) -> str:
        lines = [
            f"{self.total_images} images from {len(self.per_source)} sources",
            f"  duplicates removed: {self.duplicates_removed}",
            "  splits: " + ", ".join(f"{name}={n}" for name, n in sorted(self.split_sizes.items())),
            "  by local relevance:",
        ]
        for relevance, count in sorted(self.per_relevance.items(), key=lambda x: -x[1]):
            lines.append(f"    {count:6d}  {relevance}")
        lines.append("  by class:")
        for label, count in sorted(self.per_label.items(), key=lambda x: -x[1]):
            tag = relevance_of(label).value
            lines.append(f"    {count:6d}  {label}  [{tag}]")
        return "\n".join(lines)


def collect_samples(
    *,
    data_dir: Path | None = None,
    include_augmented: bool = False,
) -> list[LeafSample]:
    """Gather samples from every downloaded catalogue entry.

    Sources that were never downloaded are skipped rather than raising, so the
    pipeline still runs with a partial download.
    """
    root = data_dir or RAW_DATA_DIR
    samples: list[LeafSample] = []

    for source in CATALOGUE:
        directory = root / source.key
        if not directory.exists():
            continue
        samples.extend(discover_samples(directory, source.key, include_augmented=include_augmented))

    if not samples:
        raise FileNotFoundError(
            f"no images found under {root}. Run `citrus-scout data download` first."
        )
    return samples


def build_splits(
    *,
    data_dir: Path | None = None,
    include_augmented: bool = False,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    min_samples_per_class: int = 20,
) -> tuple[SplitResult, DatasetSummary]:
    """Collect, filter, split and verify the combined dataset.

    Classes below `min_samples_per_class` are dropped: with a handful of images they
    cannot support a meaningful validation estimate, and they distort the macro
    averages that would otherwise summarise the model.
    """
    samples = collect_samples(data_dir=data_dir, include_augmented=include_augmented)

    counts = Counter(s.label for s in samples)
    kept = [s for s in samples if counts[s.label] >= min_samples_per_class]
    if not kept:
        raise ValueError(
            f"every class has fewer than {min_samples_per_class} samples; "
            "lower min_samples_per_class or download more data"
        )

    split = stratified_split(
        kept,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        deduplicate_first=True,
    )
    verify_no_leakage(split)

    surviving = split.train + split.val + split.test
    per_relevance: Counter[str] = Counter()
    for sample in surviving:
        per_relevance[relevance_of(sample.label).value] += 1

    summary = DatasetSummary(
        total_images=len(surviving),
        per_source=dict(Counter(s.source for s in surviving)),
        per_label=dict(Counter(s.label for s in surviving)),
        per_relevance=dict(per_relevance),
        duplicates_removed=split.duplicates_removed,
        split_sizes={
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
    )
    return split, summary


def binary_labels(samples: list[LeafSample]) -> list[int]:
    """Collapse multi-class labels into healthy (0) versus affected (1).

    This is the task the product actually performs: the technician needs to know
    which trees to visit, not the exact pathogen. Naming the pathogen is a separate,
    harder problem, and for classes absent from Spain it is one we cannot honestly
    claim to solve.
    """
    return [0 if relevance_of(s.label) is LocalRelevance.HEALTHY else 1 for s in samples]
