"""Leaf image dataset assembled from the downloaded public sources.

The datasets in the catalogue disagree about layout: one nests train/test splits
inside the archive, another groups by class at the top level, and one ships both a
raw and a pre-augmented copy of the same photographs. Rather than trusting any of
those structures, we walk the tree, read the class from the parent folder, and
build our own splits.

That last point matters more than it looks. Two failure modes here would silently
inflate every metric we report:

1. **Augmented duplicates.** If a pre-augmented copy of a leaf lands in train while
   the original lands in validation, the model is being tested on data it has seen.
   Directories that look augmented are excluded by default.

2. **Naive splitting.** Several of these datasets photograph the same leaf multiple
   times. A random split scatters those near-duplicates across train and validation.
   We cannot fully detect that without perceptual hashing, so this is called out in
   the report rather than silently ignored.
"""

from __future__ import annotations

import contextlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Directory names that indicate a pre-augmented copy rather than original imagery.
AUGMENTED_DIR_PATTERN = re.compile(r"(^|[-_ ])aug(mented)?([-_ ]|$)", re.IGNORECASE)

# Structural directory names that are not class labels.
SPLIT_DIR_NAMES = frozenset({"train", "test", "val", "valid", "validation", "eval"})


@dataclass(frozen=True)
class LeafSample:
    """One labelled image on disk."""

    path: Path
    label: str
    source: str
    """Catalogue key of the dataset this came from."""


def looks_augmented(path: Path, root: Path | None = None) -> bool:
    """Whether any directory inside the dataset marks this as pre-augmented data.

    Only the portion of the path below `root` is inspected. Checking the absolute
    path instead would let a directory anywhere above the dataset (a home folder, a
    CI workspace, a pytest temporary directory) match the pattern and silently
    discard every image in the dataset.
    """
    parts = path.parts
    if root is not None:
        # Not under root: fall back to inspecting the whole path.
        with contextlib.suppress(ValueError):
            parts = path.relative_to(root).parts
    # Drop the filename: only directory names denote an augmented copy.
    return any(AUGMENTED_DIR_PATTERN.search(part) for part in parts[:-1])


def infer_label(path: Path, root: Path) -> str | None:
    """Read the class label from the first meaningful parent folder.

    Walks up from the file, skipping split directories (train/test/val), and returns
    the first directory that names a class. Returns None when the file sits directly
    under the dataset root with no class folder.
    """
    relative = path.relative_to(root)
    # Drop the filename, then walk upwards looking for a non-structural name.
    for part in reversed(relative.parts[:-1]):
        if part.lower() in SPLIT_DIR_NAMES:
            continue
        return part
    return None


def normalise_label(label: str) -> str:
    """Canonicalise a class name so the same class from two datasets agrees.

    'Citrus Canker', 'citrus_canker' and 'citrus-canker' all become 'citrus canker'.
    """
    cleaned = re.sub(r"[-_]+", " ", label).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def discover_samples(
    root: Path,
    source_key: str,
    *,
    include_augmented: bool = False,
) -> list[LeafSample]:
    """Walk a downloaded dataset directory and collect labelled images."""
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    samples: list[LeafSample] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not include_augmented and looks_augmented(path, root):
            continue
        label = infer_label(path, root)
        if label is None:
            continue
        samples.append(LeafSample(path=path, label=normalise_label(label), source=source_key))
    return samples


class LeafDataset(Dataset):
    """Torch dataset over discovered leaf images.

    Labels are indices into `classes`, which is sorted so the mapping is stable
    across runs and machines. That stability matters: a checkpoint trained on one
    ordering produces silently wrong predictions when loaded against another.
    """

    def __init__(
        self,
        samples: Sequence[LeafSample],
        *,
        classes: Sequence[str] | None = None,
        transform: Callable | None = None,
    ) -> None:
        if not samples:
            raise ValueError("cannot build a dataset from zero samples")

        self.samples = list(samples)
        self.transform = transform
        self.classes = list(classes) if classes else sorted({s.label for s in self.samples})
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}

        unknown = {s.label for s in self.samples} - set(self.class_to_index)
        if unknown:
            raise ValueError(f"samples carry labels missing from `classes`: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        # Convert to RGB: some of these files are palettised PNG or CMYK JPEG, which
        # would otherwise yield tensors with the wrong channel count.
        image = Image.open(sample.path).convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]

        return image, self.class_to_index[sample.label]

    def class_counts(self) -> Counter:
        """Number of samples per class label."""
        return Counter(s.label for s in self.samples)

    def __repr__(self) -> str:
        return (
            f"LeafDataset(n={len(self)}, classes={len(self.classes)}, "
            f"sources={sorted({s.source for s in self.samples})})"
        )
