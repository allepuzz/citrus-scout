"""Tests for Grad-CAM attention maps.

Two things carry real risk here. Hooks that leak keep every activation alive and look
like an unrelated memory problem, so they are tested for removal. And the
background-shortcut metric has to mean the same thing at any feature-map size, since
EfficientNet-B0 produces 7x7 maps where a one-cell ring is already half the area.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from citrus_scout.evaluation.gradcam import (
    GradCAM,
    denormalise,
    overlay,
    resolve_target_layer,
    summarise_attention,
)


class SmallConvNet(nn.Module):
    """Minimal conv net with a named final layer, for fast hook tests."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.final_conv(self.features(x))
        return self.classifier(self.pool(x).flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SmallConvNet()


@pytest.fixture
def images():
    torch.manual_seed(1)
    return torch.randn(2, 3, 32, 32)


class TestResolveTargetLayer:
    def test_finds_the_efficientnet_layer(self):
        from citrus_scout.models.classifier import build_model

        efficientnet = build_model(num_classes=2, pretrained=False, dropout=0.0)
        assert resolve_target_layer(efficientnet, "efficientnet_b0") == "bn2"

    def test_rejects_an_unknown_family_rather_than_guessing(self):
        # A silently wrong layer yields a plausible heatmap of nothing in
        # particular, which invites a confident wrong conclusion.
        with pytest.raises(ValueError, match="no default Grad-CAM layer"):
            resolve_target_layer(SmallConvNet(), "some_new_architecture")

    def test_reports_when_the_expected_layer_is_missing(self):
        # A timm release renaming bn2 should fail loudly here.
        with pytest.raises(ValueError, match="has no layer"):
            resolve_target_layer(SmallConvNet(), "efficientnet_b0")


class TestGradCAMHooks:
    def test_hooks_are_removed_on_exit(self):
        # A leaked hook pins activations for the life of the process.
        model = SmallConvNet()
        before = len(model.final_conv._forward_hooks)

        with GradCAM(model, "final_conv") as cam:
            assert len(model.final_conv._forward_hooks) == before + 1
            assert cam._handles

        assert len(model.final_conv._forward_hooks) == before
        assert not cam._handles

    def test_hooks_are_removed_even_when_the_body_raises(self):
        model = SmallConvNet()
        before = len(model.final_conv._forward_hooks)

        with pytest.raises(RuntimeError, match="deliberate"), GradCAM(model, "final_conv"):
            raise RuntimeError("deliberate")

        assert len(model.final_conv._forward_hooks) == before

    def test_cached_tensors_are_released_on_exit(self, model, images):
        with GradCAM(model, "final_conv") as cam:
            cam.heatmap(images)
            assert cam._activations is not None
        assert cam._activations is None
        assert cam._gradients is None

    def test_refuses_to_run_without_the_context_manager(self, model, images):
        cam = GradCAM(model, "final_conv")
        with pytest.raises(RuntimeError, match="context manager"):
            cam.heatmap(images)

    def test_rejects_an_unknown_layer_name(self, model):
        with pytest.raises(ValueError, match="not found"):
            GradCAM(model, "no_such_layer")


class TestHeatmap:
    def test_one_map_per_image(self, model, images):
        with GradCAM(model, "final_conv") as cam:
            maps = cam.heatmap(images)
        assert maps.shape[0] == images.shape[0]
        assert maps.ndim == 3

    def test_each_map_is_normalised_independently(self, model, images):
        # Per image, not per batch: gradient magnitude scales with confidence, so a
        # batch-wide normalisation would make confident images look more attentive.
        with GradCAM(model, "final_conv") as cam:
            maps = cam.heatmap(images)
        for single in maps:
            assert single.min() == pytest.approx(0.0, abs=1e-6)
            assert single.max() == pytest.approx(1.0, abs=1e-6)

    def test_no_negative_values(self, model, images):
        # The ReLU keeps only evidence *for* the class.
        with GradCAM(model, "final_conv") as cam:
            maps = cam.heatmap(images)
        assert maps.min() >= 0.0

    def test_an_explicit_target_class_changes_the_map(self, model, images):
        with GradCAM(model, "final_conv") as cam:
            for_zero = cam.heatmap(images, target_class=0)
            for_one = cam.heatmap(images, target_class=1)
        assert not np.allclose(for_zero, for_one)

    def test_model_is_left_in_eval_mode(self, model, images):
        model.eval()
        with GradCAM(model, "final_conv") as cam:
            cam.heatmap(images)
        assert not model.training

    def test_training_mode_is_restored(self, model, images):
        # The method flips to eval internally; a caller who was training should get
        # that state back rather than a silently switched model.
        model.train()
        with GradCAM(model, "final_conv") as cam:
            cam.heatmap(images)
        assert model.training

    def test_leaves_no_gradients_on_the_parameters(self, model, images):
        # Grad-CAM backpropagates. Leaving those gradients behind would corrupt the
        # next optimiser step if this ran mid-training.
        with GradCAM(model, "final_conv") as cam:
            cam.heatmap(images)
        assert all(p.grad is None for p in model.parameters())

    @pytest.mark.slow
    def test_works_on_the_real_backbone(self):
        from citrus_scout.models.classifier import build_model

        torch.manual_seed(0)
        efficientnet = build_model(num_classes=2, pretrained=False, dropout=0.3)
        layer = resolve_target_layer(efficientnet, "efficientnet_b0")

        with GradCAM(efficientnet, layer) as cam:
            maps = cam.heatmap(torch.randn(2, 3, 224, 224))

        # EfficientNet-B0 reduces 224x224 to a 7x7 feature map.
        assert maps.shape == (2, 7, 7)
        assert maps.min() >= 0.0


class TestSummariseAttention:
    def test_uniform_attention_is_exactly_proportional(self):
        # The property that makes the metric readable: a flat map scores 1.0, at
        # every resolution, so the threshold means one thing everywhere.
        for size in (7, 14, 28, 224):
            summary = summarise_attention(np.ones((size, size)))
            assert summary.border_excess == pytest.approx(1.0, abs=1e-9)
            assert not summary.looks_like_background_shortcut()

    def test_centred_attention_is_clean(self):
        heatmap = np.zeros((7, 7))
        heatmap[2:5, 2:5] = 1.0
        summary = summarise_attention(heatmap)
        assert summary.border_fraction == pytest.approx(0.0)
        assert not summary.looks_like_background_shortcut()

    def test_border_attention_is_flagged(self):
        # The shortcut this module exists to catch: all the evidence at the frame
        # edge, where a leaf photograph holds no diagnostic information.
        heatmap = np.ones((7, 7))
        heatmap[1:6, 1:6] = 0.0
        summary = summarise_attention(heatmap)
        assert summary.border_excess > 2.0
        assert summary.looks_like_background_shortcut()

    def test_border_verdict_is_resolution_independent(self):
        # The bug this guards: with a fixed threshold on the raw fraction, the same
        # attention pattern was suspicious at 7x7 and clean at 14x14.
        verdicts = []
        for size in (7, 14, 28, 56):
            heatmap = np.ones((size, size))
            margin = max(int(size * 0.125), 1)
            heatmap[margin:-margin, margin:-margin] = 0.0
            verdicts.append(summarise_attention(heatmap).looks_like_background_shortcut())
        assert all(verdicts), "an all-border map must be flagged at every size"

    def test_centre_fraction_measures_the_middle(self):
        heatmap = np.zeros((8, 8))
        heatmap[2:6, 2:6] = 1.0
        assert summarise_attention(heatmap).centre_fraction == pytest.approx(1.0)

    def test_concentration_is_high_for_a_single_peak(self):
        heatmap = np.zeros((10, 10))
        heatmap[5, 5] = 1.0
        assert summarise_attention(heatmap).concentration == pytest.approx(1.0)

    def test_concentration_is_low_for_a_flat_map(self):
        # A flat map puts exactly its area share in the top decile and explains
        # nothing about where the model looked.
        assert summarise_attention(np.ones((10, 10))).concentration == pytest.approx(0.1, abs=0.01)

    def test_records_the_peak(self):
        heatmap = np.zeros((5, 5))
        heatmap[2, 2] = 0.73
        assert summarise_attention(heatmap).peak_value == pytest.approx(0.73)

    def test_an_all_zero_map_does_not_divide_by_zero(self):
        summary = summarise_attention(np.zeros((7, 7)))
        assert summary.border_fraction == 0.0
        assert summary.border_excess == pytest.approx(0.0)

    def test_rejects_a_batch(self):
        with pytest.raises(ValueError, match="single 2D heatmap"):
            summarise_attention(np.ones((2, 7, 7)))

    def test_rejects_an_impossible_border_width(self):
        with pytest.raises(ValueError, match="border_width must be"):
            summarise_attention(np.ones((7, 7)), border_width=0.6)


class TestOverlay:
    def test_denormalise_returns_displayable_pixels(self):
        restored = denormalise(torch.zeros(3, 16, 16))
        assert restored.dtype == np.uint8
        assert restored.shape == (16, 16, 3)

    def test_denormalise_inverts_the_normalisation(self):
        from citrus_scout.data.transforms import IMAGENET_MEAN, IMAGENET_STD

        # A mid-grey image, normalised, must come back as mid-grey.
        grey = torch.full((3, 8, 8), 0.5)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        normalised = (grey - mean) / std

        restored = denormalise(normalised)
        assert restored.mean() == pytest.approx(127.5, abs=2)

    def test_overlay_matches_the_image_size(self):
        # The heatmap is 7x7 and must be upsampled to the image, not the reverse.
        blended = overlay(torch.zeros(3, 224, 224), np.ones((7, 7)))
        assert blended.shape == (224, 224, 3)
        assert blended.dtype == np.uint8

    def test_overlay_stays_in_range(self):
        torch.manual_seed(0)
        blended = overlay(torch.randn(3, 64, 64), np.random.default_rng(0).random((7, 7)))
        assert blended.min() >= 0
        assert blended.max() <= 255

    def test_zero_alpha_leaves_the_image_untouched(self):
        image = torch.zeros(3, 32, 32)
        assert np.array_equal(overlay(image, np.ones((7, 7)), alpha=0.0), denormalise(image))

    def test_rejects_an_out_of_range_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            overlay(torch.zeros(3, 8, 8), np.ones((7, 7)), alpha=1.5)


class TestShortcutDetection:
    """End to end: does the detector catch a model that reads the background?

    Two datasets where the class is encoded in one region only. Both train to 100%
    accuracy, so no metric in this project can separate them; the heatmap can.
    """

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.final_conv = nn.Conv2d(32, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(32, 2)

        def forward(self, x):
            return self.head(self.pool(self.final_conv(self.body(x))).flatten(1))

    @staticmethod
    def _data(signal_at: str, *, n: int = 64, size: int = 32):
        generator = torch.Generator().manual_seed(0)
        images = torch.randn(n, 3, size, size, generator=generator) * 0.3
        labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()

        band = size // 8
        for index in range(n):
            if labels[index] == 0:
                continue
            if signal_at == "border":
                images[index, :, :band, :] += 3.0
                images[index, :, -band:, :] += 3.0
                images[index, :, :, :band] += 3.0
                images[index, :, :, -band:] += 3.0
            else:
                low, high = size // 2 - band, size // 2 + band
                images[index, :, low:high, low:high] += 3.0

        return images, labels

    def _train_and_summarise(self, signal_at: str):
        from torch.utils.data import DataLoader, TensorDataset

        images, labels = self._data(signal_at)

        torch.manual_seed(0)
        model = self._Net()
        loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(25):
            for batch_images, batch_labels in loader:
                optimizer.zero_grad()
                criterion(model(batch_images), batch_labels).backward()
                optimizer.step()
        model.eval()

        with torch.no_grad():
            accuracy = (model(images).argmax(1) == labels).float().mean().item()

        summaries = []
        with GradCAM(model, "final_conv") as cam:
            for single in cam.heatmap(images):
                summaries.append(summarise_attention(single))

        return accuracy, summaries

    @pytest.mark.slow
    def test_a_border_reading_model_is_flagged(self):
        accuracy, summaries = self._train_and_summarise("border")
        excess = np.median([s.border_excess for s in summaries])

        assert accuracy > 0.95, "the shortcut model should fit the data easily"
        assert excess > 1.15, f"border excess {excess:.2f} should exceed the threshold"

    @pytest.mark.slow
    def test_a_centre_reading_model_is_not_flagged(self):
        # The control. Without this, a detector that flags everything would pass.
        accuracy, summaries = self._train_and_summarise("centre")
        excess = np.median([s.border_excess for s in summaries])

        assert accuracy > 0.95
        assert excess < 1.0, f"border excess {excess:.2f} should stay below proportional"
        assert not any(s.looks_like_background_shortcut() for s in summaries)
