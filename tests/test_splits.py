"""Tests for dataset splitting and leakage prevention."""

from __future__ import annotations

from pathlib import Path

import pytest

from citrus_scout.data.leaf_dataset import LeafSample
from citrus_scout.data.splits import (
    SplitResult,
    deduplicate,
    file_digest,
    stratified_split,
    verify_no_leakage,
)


def make_samples(label: str, count: int, *, source: str = "test") -> list[LeafSample]:
    """Build synthetic samples with distinct paths."""
    return [
        LeafSample(path=Path(f"/data/{label}/{i:04d}.jpg"), label=label, source=source)
        for i in range(count)
    ]


class TestStratifiedSplit:
    def test_every_class_appears_in_every_split(self):
        samples = make_samples("healthy", 100) + make_samples("canker", 60)
        split = stratified_split(samples, deduplicate_first=False)

        for group in (split.train, split.val, split.test):
            assert {s.label for s in group} == {"healthy", "canker"}

    def test_no_sample_is_lost(self):
        samples = make_samples("healthy", 50) + make_samples("mite", 30)
        split = stratified_split(samples, deduplicate_first=False)

        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(samples)

    def test_fractions_are_respected_per_class(self):
        samples = make_samples("healthy", 200)
        split = stratified_split(
            samples, val_fraction=0.2, test_fraction=0.1, deduplicate_first=False
        )

        assert len(split.val) == pytest.approx(40, abs=1)
        assert len(split.test) == pytest.approx(20, abs=1)

    def test_same_seed_gives_same_split(self):
        samples = make_samples("healthy", 80) + make_samples("canker", 40)
        first = stratified_split(samples, seed=7, deduplicate_first=False)
        second = stratified_split(samples, seed=7, deduplicate_first=False)

        assert [s.path for s in first.train] == [s.path for s in second.train]

    def test_different_seed_gives_different_split(self):
        samples = make_samples("healthy", 100)
        first = stratified_split(samples, seed=1, deduplicate_first=False)
        second = stratified_split(samples, seed=2, deduplicate_first=False)

        assert [s.path for s in first.train] != [s.path for s in second.train]

    def test_tiny_class_keeps_a_training_sample(self):
        """A class with very few images must not be split entirely into held-out sets."""
        samples = make_samples("healthy", 100) + make_samples("rare", 3)
        split = stratified_split(samples, deduplicate_first=False)

        assert any(s.label == "rare" for s in split.train)

    def test_single_sample_class_goes_to_train(self):
        samples = make_samples("healthy", 50) + make_samples("singleton", 1)
        split = stratified_split(samples, deduplicate_first=False)

        assert sum(1 for s in split.train if s.label == "singleton") == 1

    @pytest.mark.parametrize(
        ("val_fraction", "test_fraction"),
        [(0.0, 0.0), (0.6, 0.5), (-0.1, 0.2)],
    )
    def test_invalid_fractions(self, val_fraction, test_fraction):
        samples = make_samples("healthy", 10)
        with pytest.raises(ValueError, match="fraction"):
            stratified_split(
                samples,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                deduplicate_first=False,
            )

    def test_empty_input(self):
        with pytest.raises(ValueError, match="no samples"):
            stratified_split([], deduplicate_first=False)


class TestLeakage:
    def test_clean_split_passes(self):
        samples = make_samples("healthy", 60) + make_samples("canker", 40)
        split = stratified_split(samples, deduplicate_first=False)
        verify_no_leakage(split)

    def test_overlap_is_detected(self):
        shared = make_samples("healthy", 5)
        split = SplitResult(
            train=shared,
            val=shared,  # same objects in both splits
            test=make_samples("canker", 5),
            duplicates_removed=0,
        )
        with pytest.raises(AssertionError, match="both train and val"):
            verify_no_leakage(split)

    def test_real_split_has_no_leakage(self):
        """Splitting many classes at once must not scatter a sample into two splits."""
        samples = []
        for label, count in [("healthy", 300), ("canker", 120), ("mite", 45), ("curl", 12)]:
            samples.extend(make_samples(label, count))

        split = stratified_split(samples, seed=123, deduplicate_first=False)
        verify_no_leakage(split)


class TestDeduplication:
    def test_identical_files_are_collapsed(self, tmp_path):
        content = b"same image bytes"
        paths = []
        for i in range(3):
            p = tmp_path / f"copy_{i}.jpg"
            p.write_bytes(content)
            paths.append(p)

        samples = [LeafSample(path=p, label="healthy", source="t") for p in paths]
        kept, removed = deduplicate(samples)

        assert len(kept) == 1
        assert removed == 2

    def test_distinct_files_are_kept(self, tmp_path):
        samples = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            p.write_bytes(f"content {i}".encode())
            samples.append(LeafSample(path=p, label="healthy", source="t"))

        kept, removed = deduplicate(samples)

        assert len(kept) == 3
        assert removed == 0

    def test_duplicate_across_classes_is_removed(self, tmp_path):
        """The same photograph filed under two classes is a labelling error.

        Keeping both would put contradictory labels on identical pixels, and if they
        land in different splits the model is evaluated on memorised data.
        """
        content = b"ambiguous leaf"
        first = tmp_path / "healthy.jpg"
        second = tmp_path / "canker.jpg"
        first.write_bytes(content)
        second.write_bytes(content)

        samples = [
            LeafSample(path=first, label="healthy", source="t"),
            LeafSample(path=second, label="canker", source="t"),
        ]
        kept, removed = deduplicate(samples)

        assert len(kept) == 1
        assert removed == 1

    def test_missing_file_is_dropped_not_fatal(self, tmp_path):
        good = tmp_path / "good.jpg"
        good.write_bytes(b"fine")

        samples = [
            LeafSample(path=good, label="healthy", source="t"),
            LeafSample(path=tmp_path / "gone.jpg", label="healthy", source="t"),
        ]
        kept, removed = deduplicate(samples)

        assert len(kept) == 1
        assert removed == 1

    def test_split_reports_duplicates_removed(self, tmp_path):
        content = b"dup"
        samples = []
        for i in range(4):
            p = tmp_path / f"d{i}.jpg"
            p.write_bytes(content if i < 2 else f"unique {i}".encode())
            samples.append(LeafSample(path=p, label="healthy", source="t"))

        split = stratified_split(samples, deduplicate_first=True)
        assert split.duplicates_removed == 1


class TestFileDigest:
    def test_same_content_same_digest(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")

        assert file_digest(a) == file_digest(b)

    def test_different_content_different_digest(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"one")
        b.write_bytes(b"two")

        assert file_digest(a) != file_digest(b)

    def test_large_file_is_chunked(self, tmp_path):
        """Digest must not depend on the chunk size used to read the file."""
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (3 * 1024 * 1024))

        assert file_digest(big, chunk_size=1024) == file_digest(big, chunk_size=1 << 20)
