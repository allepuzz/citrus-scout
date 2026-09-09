"""Read a packaged dataset archive.

This is the path used on Colab. The archive carries its own split assignment, so a
remote run trains on exactly the partition the local machine produced, rather than
recomputing one that might differ.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from citrus_scout.data.taxonomy import is_healthy

SPLIT_NAMES = ("train", "val", "test")


class ArchiveDataset(Dataset):
    """Dataset backed by an extracted archive directory.

    Layout is `<root>/<split>/<label>/<index>.jpg`, as written by
    `citrus_scout.data.package`.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        classes: list[str],
        transform: Callable | None = None,
        binary: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.binary = binary

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split directory not found: {split_dir}")

        self.paths: list[Path] = []
        self.labels: list[str] = []
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for image_path in sorted(label_dir.glob("*.jpg")):
                self.paths.append(image_path)
                self.labels.append(label_dir.name)

        if not self.paths:
            raise ValueError(f"no images found under {split_dir}")

        self.classes = ["healthy", "affected"] if binary else list(classes)
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}

    def _target(self, label: str) -> int:
        if self.binary:
            return 0 if is_healthy(label) else 1
        return self.class_to_index[label]

    def label_pairs(self) -> list[tuple[str, int]]:
        """Original label and its integer target, for class weighting."""
        return [(label, self._target(label)) for label in self.labels]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
        return image, self._target(self.labels[index])


def extract_archive(archive: Path, destination: Path | None = None) -> Path:
    """Unpack an archive, skipping the work when it is already extracted.

    Returns the directory holding the split folders.
    """
    archive = Path(archive)
    target = destination or archive.with_suffix("")

    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
    return target


def read_manifest(root: Path) -> dict:
    """Read the manifest written alongside the images."""
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_packaged_splits(
    archive: Path,
    *,
    train_tf: Callable | None = None,
    eval_tf: Callable | None = None,
    binary: bool = True,
    destination: Path | None = None,
) -> tuple[ArchiveDataset, ArchiveDataset, ArchiveDataset, list[str]]:
    """Extract an archive and build the three datasets from it."""
    root = extract_archive(Path(archive), destination)
    manifest = read_manifest(root)
    classes = list(manifest["classes"])

    datasets = tuple(
        ArchiveDataset(
            root,
            split,
            classes=classes,
            transform=train_tf if split == "train" else eval_tf,
            binary=binary,
        )
        for split in SPLIT_NAMES
    )
    train_ds, val_ds, test_ds = datasets
    return train_ds, val_ds, test_ds, list(train_ds.classes)
