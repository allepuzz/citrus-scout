"""Train/validation/test splitting.

Splitting is where imbalanced image datasets quietly go wrong, so the choices here
are deliberate:

**Stratified.** With classes ranging from a few dozen to several hundred images, a
random split can leave a rare class with almost nothing in validation, making its
metrics meaningless noise.

**Deduplicated by content.** Several of these datasets contain byte-identical
copies of the same photograph, sometimes across different class folders. If a
duplicate straddles the train and validation split, the model is evaluated on data
it memorised, and every metric we report is inflated. Hashing file contents catches
exact duplicates; it does not catch a re-encoded or resized copy of the same photo,
which is a known remaining risk.

**Seeded.** The same seed must give the same split on every machine, otherwise a
result reported from Colab cannot be reproduced locally.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from citrus_scout.data.leaf_dataset import LeafSample


@dataclass(frozen=True)
class SplitResult:
    """Train/validation/test partition of a sample list."""

    train: list[LeafSample]
    val: list[LeafSample]
    test: list[LeafSample]
    duplicates_removed: int

    def __repr__(self) -> str:
        return (
            f"SplitResult(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)}, duplicates_removed={self.duplicates_removed})"
        )


def file_digest(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, read in chunks to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(samples: Sequence[LeafSample]) -> tuple[list[LeafSample], int]:
    """Drop byte-identical images, keeping the first occurrence.

    Returns the surviving samples and how many were removed. Only exact duplicates
    are caught; a resized or re-encoded copy of the same photograph survives this
    and remains a source of optimistic bias.
    """
    seen: dict[str, LeafSample] = {}
    removed = 0

    for sample in samples:
        try:
            digest = file_digest(sample.path)
        except OSError:
            # Unreadable file: drop it rather than crashing a long pipeline run.
            removed += 1
            continue

        if digest in seen:
            removed += 1
            continue
        seen[digest] = sample

    return list(seen.values()), removed


def stratified_split(
    samples: Sequence[LeafSample],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    deduplicate_first: bool = True,
) -> SplitResult:
    """Partition samples per class, so every class appears in every split.

    Args:
        samples: labelled images to partition.
        val_fraction: share of each class held out for validation.
        test_fraction: share of each class held out for test.
        seed: controls the shuffle; the same seed gives the same split anywhere.
        deduplicate_first: drop byte-identical images before splitting.
    """
    # Check each fraction on its own before checking the sum: a negative value can
    # otherwise be masked by a larger positive one and produce a silently empty split.
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError(
            f"fractions must not be negative, got val={val_fraction} test={test_fraction}"
        )
    if not 0.0 < val_fraction + test_fraction < 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be in (0, 1), got {val_fraction} + {test_fraction}"
        )

    working = list(samples)
    removed = 0
    if deduplicate_first:
        working, removed = deduplicate(working)

    if not working:
        raise ValueError("no samples left to split")

    by_class: dict[str, list[LeafSample]] = defaultdict(list)
    for sample in working:
        by_class[sample.label].append(sample)

    rng = np.random.default_rng(seed)
    train: list[LeafSample] = []
    val: list[LeafSample] = []
    test: list[LeafSample] = []

    for label in sorted(by_class):
        group = by_class[label]
        # Sort by path first so the shuffle starts from a deterministic order
        # regardless of how the filesystem enumerated the files.
        group.sort(key=lambda s: str(s.path))
        indices = rng.permutation(len(group))

        n_val = round(len(group) * val_fraction)
        n_test = round(len(group) * test_fraction)

        # Guarantee at least one sample in train for any class we keep at all,
        # otherwise a tiny class can be split entirely into held-out sets.
        while n_val + n_test >= len(group) and (n_val > 0 or n_test > 0):
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1

        val.extend(group[i] for i in indices[:n_val])
        test.extend(group[i] for i in indices[n_val : n_val + n_test])
        train.extend(group[i] for i in indices[n_val + n_test :])

    return SplitResult(train=train, val=val, test=test, duplicates_removed=removed)


def verify_no_leakage(split: SplitResult) -> None:
    """Raise if any file path appears in more than one split.

    A cheap invariant that catches indexing mistakes in the splitting code itself.
    It cannot detect two different files holding the same photograph.
    """
    partitions = {"train": split.train, "val": split.val, "test": split.test}
    paths = {name: {str(s.path) for s in group} for name, group in partitions.items()}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = paths[left] & paths[right]
        if overlap:
            example = sorted(overlap)[:3]
            raise AssertionError(
                f"{len(overlap)} samples appear in both {left} and {right}, e.g. {example}"
            )
