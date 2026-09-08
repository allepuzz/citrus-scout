"""Tests for sample discovery, label inference and the torch dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from citrus_scout.data.leaf_dataset import (
    LeafDataset,
    LeafSample,
    discover_samples,
    infer_label,
    looks_augmented,
    normalise_label,
)


def write_image(path: Path, *, size: tuple[int, int] = (16, 16), colour: str = "green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


class TestNormaliseLabel:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Citrus Canker", "citrus canker"),
            ("citrus_canker", "citrus canker"),
            ("citrus-canker", "citrus canker"),
            ("  Black   Spot  ", "black spot"),
            ("HEALTHY", "healthy"),
        ],
    )
    def test_variants_collapse(self, raw, expected):
        assert normalise_label(raw) == expected

    def test_same_class_from_two_datasets_agrees(self):
        """Datasets spell classes differently; they must land on one label."""
        assert normalise_label("Spider_Mite") == normalise_label("spider mite")


class TestLooksAugmented:
    @pytest.mark.parametrize(
        "path",
        [
            Path("data/CitrusLeafPathology-14-Aug/train/healthy/a.png"),
            Path("data/augmented/healthy/a.png"),
            Path("data/set_aug/healthy/a.png"),
            Path("data/AUGMENTED_v2/healthy/a.png"),
        ],
    )
    def test_augmented_paths_are_flagged(self, path):
        assert looks_augmented(path)

    @pytest.mark.parametrize(
        "path",
        [
            Path("data/CitrusLeafPathology-14/train/healthy/a.png"),
            Path("data/raw/healthy/a.png"),
            Path("data/august_harvest/healthy/a.png"),  # 'august', not 'aug'
        ],
    )
    def test_ordinary_paths_are_not_flagged(self, path):
        assert not looks_augmented(path)

    def test_directory_above_root_is_ignored(self):
        """A parent directory matching the pattern must not condemn the dataset.

        Without a root, a checkout under ~/augmented-projects/ or a CI workspace
        containing 'aug' would discard every image silently.
        """
        root = Path("/home/user/augmented-projects/citrus/data/raw")
        image = root / "healthy" / "a.png"

        assert looks_augmented(image)  # absolute path does match
        assert not looks_augmented(image, root)  # relative to root, it does not

    def test_augmented_below_root_is_still_flagged(self):
        root = Path("/home/user/citrus/data/raw")
        image = root / "set-Aug" / "healthy" / "a.png"

        assert looks_augmented(image, root)

    def test_filename_alone_does_not_flag(self):
        """Only directories denote an augmented copy, not the file's own name."""
        root = Path("/data/raw")
        assert not looks_augmented(root / "healthy" / "aug.png", root)


class TestInferLabel:
    def test_reads_parent_folder(self, tmp_path):
        img = tmp_path / "healthy" / "a.jpg"
        assert infer_label(img, tmp_path) == "healthy"

    def test_skips_split_directories(self, tmp_path):
        """train/test/val name a split, not a class."""
        img = tmp_path / "train" / "citrus canker" / "a.jpg"
        assert infer_label(img, tmp_path) == "citrus canker"

    def test_skips_nested_split_directory(self, tmp_path):
        img = tmp_path / "dataset" / "test" / "healthy" / "a.jpg"
        assert infer_label(img, tmp_path) == "healthy"

    def test_returns_none_without_class_folder(self, tmp_path):
        img = tmp_path / "a.jpg"
        assert infer_label(img, tmp_path) is None


class TestDiscoverSamples:
    def test_finds_labelled_images(self, tmp_path):
        write_image(tmp_path / "healthy" / "a.jpg")
        write_image(tmp_path / "healthy" / "b.jpg")
        write_image(tmp_path / "canker" / "c.jpg")

        samples = discover_samples(tmp_path, "src")

        assert len(samples) == 3
        assert {s.label for s in samples} == {"healthy", "canker"}

    def test_excludes_augmented_by_default(self, tmp_path):
        write_image(tmp_path / "raw_set" / "healthy" / "a.jpg")
        write_image(tmp_path / "set-Aug" / "healthy" / "b.jpg")

        assert len(discover_samples(tmp_path, "src")) == 1
        assert len(discover_samples(tmp_path, "src", include_augmented=True)) == 2

    def test_ignores_non_images(self, tmp_path):
        write_image(tmp_path / "healthy" / "a.jpg")
        (tmp_path / "healthy" / "notes.txt").write_text("ignore me")
        (tmp_path / "healthy" / "data.csv").write_text("a,b")

        assert len(discover_samples(tmp_path, "src")) == 1

    def test_records_source_key(self, tmp_path):
        write_image(tmp_path / "healthy" / "a.jpg")
        samples = discover_samples(tmp_path, "my_dataset")
        assert samples[0].source == "my_dataset"

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_samples(tmp_path / "nope", "src")

    def test_is_deterministic(self, tmp_path):
        for i in range(5):
            write_image(tmp_path / "healthy" / f"{i}.jpg")

        first = [s.path for s in discover_samples(tmp_path, "src")]
        second = [s.path for s in discover_samples(tmp_path, "src")]
        assert first == second


class TestLeafDataset:
    @pytest.fixture
    def samples(self, tmp_path):
        paths = [
            write_image(tmp_path / "healthy" / "a.jpg"),
            write_image(tmp_path / "healthy" / "b.jpg"),
            write_image(tmp_path / "canker" / "c.jpg"),
        ]
        labels = ["healthy", "healthy", "canker"]
        return [LeafSample(path=p, label=x, source="t") for p, x in zip(paths, labels, strict=True)]

    def test_length(self, samples):
        assert len(LeafDataset(samples)) == 3

    def test_classes_are_sorted(self, samples):
        """Stable ordering: a checkpoint's class indices must mean the same thing later."""
        assert LeafDataset(samples).classes == ["canker", "healthy"]

    def test_returns_image_and_index(self, samples):
        dataset = LeafDataset(samples)
        image, label = dataset[0]

        assert isinstance(image, Image.Image)
        assert label == dataset.class_to_index[samples[0].label]

    def test_applies_transform(self, samples):
        def transform(image):
            return {"image": np.zeros((4, 4, 3), dtype=np.uint8)}

        image, _ = LeafDataset(samples, transform=transform)[0]
        assert image.shape == (4, 4, 3)

    def test_converts_to_rgb(self, tmp_path):
        """Palettised PNGs would otherwise yield the wrong channel count."""
        path = tmp_path / "healthy" / "p.png"
        path.parent.mkdir(parents=True)
        Image.new("P", (8, 8)).save(path)

        dataset = LeafDataset([LeafSample(path=path, label="healthy", source="t")])
        image, _ = dataset[0]
        assert image.mode == "RGB"

    def test_class_counts(self, samples):
        assert LeafDataset(samples).class_counts() == {"healthy": 2, "canker": 1}

    def test_explicit_class_list_is_respected(self, samples):
        dataset = LeafDataset(samples, classes=["healthy", "canker", "unseen"])
        assert dataset.class_to_index["healthy"] == 0

    def test_label_missing_from_class_list(self, samples):
        with pytest.raises(ValueError, match="missing from"):
            LeafDataset(samples, classes=["healthy"])

    def test_empty_samples(self):
        with pytest.raises(ValueError, match="zero samples"):
            LeafDataset([])
