"""Tests for configuration, packaging and the training loop's helpers."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from citrus_scout.data.archive_dataset import ArchiveDataset, extract_archive, read_manifest
from citrus_scout.data.leaf_dataset import LeafSample
from citrus_scout.data.package import _resized, package_splits
from citrus_scout.data.splits import SplitResult
from citrus_scout.data.transforms import eval_transform, train_transform
from citrus_scout.training.config import DataConfig, TrainingConfig
from citrus_scout.training.loop import class_weights, format_duration, should_stop


class TestTrainingConfig:
    @pytest.fixture
    def minimal(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"name": "test_run"}), encoding="utf-8")
        return path

    def test_loads_with_defaults(self, minimal):
        config = TrainingConfig.from_yaml(minimal)
        assert config.name == "test_run"
        assert config.model.backbone == "efficientnet_b0"

    def test_field_prevalence_defaults_to_two_percent(self, minimal):
        """The datasets are 80% affected; the field is not."""
        assert TrainingConfig.from_yaml(minimal).field_prevalence == 0.02

    def test_rejects_warmup_longer_than_training(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump({"name": "x", "optim": {"epochs": 3, "warmup_epochs": 5}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="warmup_epochs"):
            TrainingConfig.from_yaml(path)

    def test_rejects_impossible_split_fractions(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump({"name": "x", "data": {"val_fraction": 0.6, "test_fraction": 0.5}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="fractions"):
            TrainingConfig.from_yaml(path)

    def test_roundtrip_through_yaml(self, tmp_path, minimal):
        original = TrainingConfig.from_yaml(minimal)
        out = tmp_path / "written.yaml"
        original.to_yaml(out)

        assert TrainingConfig.from_yaml(out).model_dump() == original.model_dump()

    def test_rejects_non_mapping_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            TrainingConfig.from_yaml(path)

    def test_shipped_configs_are_valid(self):
        """The configs in the repo must load, or a Colab run fails at the first cell."""
        from citrus_scout.utils.paths import CONFIG_DIR

        found = list(CONFIG_DIR.glob("*.yaml"))
        assert found, "no configs found"
        for path in found:
            TrainingConfig.from_yaml(path)


class TestClassWeights:
    def test_balanced_data_gives_equal_weights(self):
        weights = class_weights([0] * 50 + [1] * 50, num_classes=2)
        assert weights[0] == pytest.approx(weights[1])

    def test_minority_class_gets_more_weight(self):
        """80% affected is the actual distribution, and it must not dominate."""
        weights = class_weights([1] * 80 + [0] * 20, num_classes=2)
        assert weights[0] > weights[1]

    def test_weights_average_to_one(self):
        """Normalisation keeps the loss scale stable when weighting is toggled."""
        weights = class_weights([1] * 80 + [0] * 20, num_classes=2)
        assert weights.mean() == pytest.approx(1.0)

    def test_absent_class_does_not_divide_by_zero(self):
        weights = class_weights([0] * 10, num_classes=2)
        assert torch.isfinite(weights).all()

    def test_weight_ratio_matches_inverse_frequency(self):
        weights = class_weights([1] * 75 + [0] * 25, num_classes=2)
        assert (weights[0] / weights[1]).item() == pytest.approx(3.0, rel=1e-5)


class TestTransforms:
    def test_train_output_shape(self):
        image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
        out = train_transform(224)(image=image)["image"]
        assert out.shape == (3, 224, 224)

    def test_eval_output_shape(self):
        image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
        out = eval_transform(224)(image=image)["image"]
        assert out.shape == (3, 224, 224)

    def test_eval_is_deterministic(self):
        """Evaluation must not be random, or results are not reproducible."""
        image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
        transform = eval_transform(224)
        first = transform(image=image)["image"]
        second = transform(image=image)["image"]
        assert torch.allclose(first, second)

    def test_train_is_random(self):
        image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
        transform = train_transform(224)
        assert not torch.allclose(transform(image=image)["image"], transform(image=image)["image"])

    def test_handles_small_image(self):
        """Some source images are smaller than the crop size."""
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        assert eval_transform(224)(image=image)["image"].shape == (3, 224, 224)


class TestResize:
    def test_downscales_large_image(self):
        assert max(_resized(Image.new("RGB", (2000, 1000)), 512).size) == 512

    def test_preserves_aspect_ratio(self):
        result = _resized(Image.new("RGB", (2000, 1000)), 512)
        assert result.size == (512, 256)

    def test_leaves_small_image_untouched(self):
        """Upscaling adds no information and costs archive space."""
        original = Image.new("RGB", (300, 200))
        assert _resized(original, 512).size == (300, 200)


class TestPackaging:
    @pytest.fixture
    def split(self, tmp_path):
        def make(label: str, count: int, prefix: str):
            out = []
            for i in range(count):
                path = tmp_path / label / f"{prefix}{i}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (800, 600), "green").save(path)
                out.append(LeafSample(path=path, label=label, source="t"))
            return out

        return SplitResult(
            train=make("healthy", 4, "tr") + make("gummosis", 3, "tr"),
            val=make("healthy", 2, "va"),
            test=make("gummosis", 2, "te"),
            duplicates_removed=7,
        )

    def test_writes_every_image(self, split, tmp_path):
        stats = package_splits(split, tmp_path / "out.zip")
        assert stats.images == 11
        assert stats.skipped == 0

    def test_manifest_records_counts(self, split, tmp_path):
        out = tmp_path / "out.zip"
        package_splits(split, out)

        with zipfile.ZipFile(out) as archive:
            manifest = json.loads(archive.read("manifest.json"))

        assert manifest["counts"] == {"train": 7, "val": 2, "test": 2}
        assert manifest["duplicates_removed"] == 7

    def test_archive_layout_is_split_then_label(self, split, tmp_path):
        out = tmp_path / "out.zip"
        package_splits(split, out)

        with zipfile.ZipFile(out) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]

        assert any(n.startswith("train/healthy/") for n in names)
        assert any(n.startswith("test/gummosis/") for n in names)

    def test_images_are_downscaled(self, split, tmp_path):
        out = tmp_path / "out.zip"
        package_splits(split, out, max_side=256)

        with zipfile.ZipFile(out) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".jpg"))
            with archive.open(name) as handle:
                assert max(Image.open(handle).size) == 256

    def test_unreadable_image_is_skipped_not_fatal(self, split, tmp_path):
        """A truncated file must not abort a long export."""
        broken = tmp_path / "healthy" / "broken.jpg"
        broken.write_bytes(b"not an image")
        split.train.append(LeafSample(path=broken, label="healthy", source="t"))

        stats = package_splits(split, tmp_path / "out.zip")
        assert stats.skipped == 1
        assert stats.images == 11


class TestArchiveDataset:
    @pytest.fixture
    def archive_root(self, tmp_path):
        root = tmp_path / "extracted"
        for split, label, count in [
            ("train", "healthy", 3),
            ("train", "gummosis", 2),
            ("val", "healthy", 1),
            ("test", "gummosis", 1),
        ]:
            directory = root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                Image.new("RGB", (256, 256), "green").save(directory / f"{i:06d}.jpg")

        manifest = {"classes": ["gummosis", "healthy"], "counts": {}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_reads_split(self, archive_root):
        dataset = ArchiveDataset(archive_root, "train", classes=["gummosis", "healthy"])
        assert len(dataset) == 5

    def test_binary_targets(self, archive_root):
        dataset = ArchiveDataset(
            archive_root, "train", classes=["gummosis", "healthy"], binary=True
        )
        targets = dict(dataset.label_pairs())
        assert targets["healthy"] == 0
        assert targets["gummosis"] == 1

    def test_multiclass_targets(self, archive_root):
        dataset = ArchiveDataset(
            archive_root, "train", classes=["gummosis", "healthy"], binary=False
        )
        targets = dict(dataset.label_pairs())
        assert targets["gummosis"] != targets["healthy"]

    def test_missing_split_is_reported(self, archive_root):
        with pytest.raises(FileNotFoundError, match="split directory"):
            ArchiveDataset(archive_root, "nonexistent", classes=[])

    def test_read_manifest(self, archive_root):
        assert read_manifest(archive_root)["classes"] == ["gummosis", "healthy"]

    def test_missing_manifest_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest"):
            read_manifest(tmp_path)

    def test_extract_is_idempotent(self, tmp_path, archive_root):
        source = tmp_path / "a.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for path in archive_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(archive_root))

        first = extract_archive(source, tmp_path / "out")
        marker = first / "extra_file.txt"
        marker.write_text("added after extraction", encoding="utf-8")

        # Second call must not re-extract and wipe local state.
        second = extract_archive(source, tmp_path / "out")
        assert second == first
        assert marker.exists()


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(45, "45s"), (90, "1m 30s"), (3661, "1h 1m")],
    )
    def test_formats(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestModel:
    """The model module had no coverage until CI caught a lint error in it.

    These tests are cheap and guard the parts that fail silently: a freeze that
    does not freeze still trains, just badly, and gives no error to notice.
    """

    def test_builds_with_requested_class_count(self):
        from citrus_scout.models.classifier import build_model

        model = build_model(num_classes=2, pretrained=False)
        output = model(torch.zeros(2, 3, 224, 224))
        assert output.shape == (2, 2)

    def test_freeze_leaves_only_the_head_trainable(self):
        from citrus_scout.models.classifier import (
            build_model,
            count_parameters,
            freeze_backbone,
        )

        model = build_model(num_classes=2, pretrained=False)
        total_before = count_parameters(model)[1]

        freeze_backbone(model)
        trainable, total = count_parameters(model)

        assert trainable < total
        assert total == total_before

    def test_unfreeze_restores_every_parameter(self):
        from citrus_scout.models.classifier import (
            build_model,
            count_parameters,
            freeze_backbone,
            unfreeze_all,
        )

        model = build_model(num_classes=2, pretrained=False)
        freeze_backbone(model)
        unfreeze_all(model)

        trainable, total = count_parameters(model)
        assert trainable == total

    def test_dropout_is_active_for_mc_dropout(self):
        """Uncertainty estimation depends on dropout actually firing.

        timm's EfficientNet applies dropout functionally from a `drop_rate`
        attribute rather than through an nn.Dropout module, so looking for the
        module finds nothing even though dropout is configured.
        """
        from citrus_scout.models.classifier import build_model, has_active_dropout

        model = build_model(num_classes=2, pretrained=False, dropout=0.3)
        model.train()
        assert has_active_dropout(model)

    def test_eval_mode_is_deterministic(self):
        from citrus_scout.models.classifier import build_model

        model = build_model(num_classes=2, pretrained=False, dropout=0.5).eval()
        batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            assert torch.allclose(model(batch), model(batch))

    def test_mc_dropout_gives_varying_predictions(self):
        """Without variation between passes there is no uncertainty to measure."""
        from citrus_scout.models.classifier import build_model, enable_mc_dropout

        model = build_model(num_classes=2, pretrained=False, dropout=0.5)
        enable_mc_dropout(model)

        batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            assert not torch.allclose(model(batch), model(batch))

    def test_mc_dropout_freezes_batchnorm_statistics(self):
        """BatchNorm must not update running stats while sampling for uncertainty."""
        from torch import nn

        from citrus_scout.models.classifier import build_model, enable_mc_dropout

        model = build_model(num_classes=2, pretrained=False, dropout=0.5)
        enable_mc_dropout(model)

        batchnorms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert batchnorms, "expected BatchNorm layers in this backbone"
        assert all(not m.training for m in batchnorms)

    def test_select_device_honours_explicit_choice(self):
        from citrus_scout.models.classifier import select_device

        assert select_device("cpu").type == "cpu"


class TestNumWorkers:
    """Worker count resolution. Colab gives 2 CPUs and torch warns when a
    DataLoader oversubscribes them, which a hardcoded 4 always did."""

    def test_explicit_value_is_honoured(self):
        config = DataConfig(num_workers=7)
        assert config.resolve_num_workers() == 7

    def test_explicit_zero_is_honoured(self):
        # 0 means "load in the main process", a legitimate request, not "unset".
        config = DataConfig(num_workers=0)
        assert config.resolve_num_workers() == 0

    def test_unset_derives_from_cpu_count(self, monkeypatch):
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: 8)
        assert DataConfig().resolve_num_workers() == 4

    def test_leaves_a_core_for_the_main_process(self, monkeypatch):
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: 3)
        assert DataConfig().resolve_num_workers() == 2

    def test_colab_two_cpus_does_not_oversubscribe(self, monkeypatch):
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: 2)
        assert DataConfig().resolve_num_workers() == 1

    def test_single_cpu_falls_back_to_main_process(self, monkeypatch):
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: 1)
        assert DataConfig().resolve_num_workers() == 0

    def test_unknown_cpu_count_does_not_crash(self, monkeypatch):
        # os.cpu_count() is documented as possibly returning None.
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: None)
        assert DataConfig().resolve_num_workers() == 0

    def test_never_negative(self, monkeypatch):
        monkeypatch.setattr("citrus_scout.training.config.os.cpu_count", lambda: 0)
        assert DataConfig().resolve_num_workers() >= 0


class TestShouldStop:
    """When to end a run.

    The bug this guards: PR-AUC is capped at 1.0, so a saturated run cannot beat
    its own best. Compared strictly, every epoch after the peak counts as a
    regression and the run dies at `patience` while still training fine.
    """

    def _call(self, pr_auc, best, stalled=0, patience=5, min_delta=1e-4):
        return should_stop(
            pr_auc=pr_auc,
            best_pr_auc=best,
            epochs_without_improvement=stalled,
            patience=patience,
            min_delta=min_delta,
        )

    def test_clear_gain_resets_the_counter(self):
        stop, stalled, _ = self._call(0.80, 0.70, stalled=3)
        assert not stop
        assert stalled == 0

    def test_no_gain_increments_the_counter(self):
        stop, stalled, _ = self._call(0.70, 0.75, stalled=2)
        assert not stop
        assert stalled == 3

    def test_stops_once_patience_is_exhausted(self):
        stop, _, reason = self._call(0.70, 0.75, stalled=4, patience=5)
        assert stop
        assert "no PR-AUC gain" in reason

    def test_gain_below_min_delta_does_not_count(self):
        # Fourth-decimal noise is not progress; without a floor patience never
        # expires on a plateau.
        stop, stalled, _ = self._call(0.9000 + 1e-6, 0.9000, stalled=1, min_delta=1e-4)
        assert not stop
        assert stalled == 2

    def test_gain_above_min_delta_counts(self):
        stop, stalled, _ = self._call(0.90 + 1e-3, 0.90, stalled=1, min_delta=1e-4)
        assert not stop
        assert stalled == 0

    def test_saturated_metric_stops_with_an_honest_reason(self):
        # The regression: a perfect score used to sit out `patience` epochs and
        # then report a plateau. It is not a plateau, it is an exhausted metric.
        stop, _, reason = self._call(1.0, 1.0, stalled=0)
        assert stop
        assert "saturated" in reason

    def test_saturation_does_not_depend_on_patience(self):
        stop, _, reason = self._call(1.0, 1.0, stalled=0, patience=None)
        assert stop
        assert "saturated" in reason

    def test_patience_none_disables_early_stopping(self):
        stop, stalled, reason = self._call(0.50, 0.90, stalled=99, patience=None)
        assert not stop
        assert reason is None
        assert stalled == 100

    def test_a_run_that_keeps_improving_is_never_stopped(self):
        # The shape of the original bug, as a sequence: monotonic gains below
        # saturation must survive any patience.
        best, stalled = -1.0, 0
        for pr_auc in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97]:
            stop, stalled, _ = self._call(pr_auc, best, stalled=stalled, patience=3)
            assert not stop, f"stopped early at {pr_auc}"
            best = max(best, pr_auc)

    def test_a_flat_run_is_stopped(self):
        best, stalled, stops = 0.90, 0, 0
        for _ in range(6):
            stop, stalled, _ = self._call(0.90, best, stalled=stalled, patience=3)
            if stop:
                stops += 1
                break
        assert stops == 1
