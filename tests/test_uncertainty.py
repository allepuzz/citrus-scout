"""Tests for MC Dropout uncertainty.

The guard that matters most: if dropout is not actually firing, every pass is
identical, variance is zero, and the system reports total confidence on every tree.
A silent failure there is worse than a crash, because the output still looks like a
valid uncertainty estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from citrus_scout.evaluation.uncertainty import (
    UncertaintyResult,
    mc_dropout_predict,
    predictive_entropy,
    uncertainty_separates_errors,
)


class TinyDropoutNet(nn.Module):
    """Smallest model that exercises the real code path.

    Uses an `nn.Dropout` module, unlike timm's EfficientNet which applies dropout
    functionally. Both paths are covered: `has_active_dropout` handles the module
    form here and the functional form in test_training.py.
    """

    def __init__(self, *, p: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.fc(self.dropout(x))


class DeterministicNet(nn.Module):
    """No dropout anywhere: the failure mode the verification guard exists to catch."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def loader():
    torch.manual_seed(0)
    features = torch.randn(32, 4)
    labels = (features[:, 0] > 0).long()
    return DataLoader(TensorDataset(features, labels), batch_size=8)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestPredictiveEntropy:
    def test_maximal_at_one_half(self):
        assert predictive_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))

    def test_zero_at_certainty(self):
        assert predictive_entropy(np.array([0.0, 1.0])) == pytest.approx([0.0, 0.0], abs=1e-9)

    def test_symmetric_about_one_half(self):
        low, high = predictive_entropy(np.array([0.2, 0.8]))
        assert low == pytest.approx(high)

    def test_no_nan_at_the_boundaries(self):
        # log(0) would produce nan; the clip is what prevents it.
        assert not np.isnan(predictive_entropy(np.array([0.0, 1.0]))).any()

    def test_monotonic_towards_one_half(self):
        values = predictive_entropy(np.array([0.01, 0.2, 0.4, 0.5]))
        assert np.all(np.diff(values) > 0)


class TestMcDropoutPredict:
    def test_produces_nonzero_spread(self, loader, device):
        # The whole point: dropout must make the passes differ.
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=10)
        assert result.std_probability.max() > 0.0

    def test_raises_when_dropout_is_inactive(self, loader, device):
        # Guards the regression fixed in d26341c: a model with no dropout would
        # otherwise report zero uncertainty everywhere and look perfectly healthy.
        with pytest.raises(RuntimeError, match="dropout is not active"):
            mc_dropout_predict(DeterministicNet(), loader, device, passes=5)

    def test_verification_can_be_waived(self, loader, device):
        # Escape hatch for deliberate deterministic runs, e.g. a timing baseline.
        result = mc_dropout_predict(
            DeterministicNet(), loader, device, passes=3, verify_dropout=False
        )
        assert result.std_probability.max() == pytest.approx(0.0, abs=1e-6)

    def test_rejects_a_single_pass(self, loader, device):
        with pytest.raises(ValueError, match="at least 2 passes"):
            mc_dropout_predict(TinyDropoutNet(), loader, device, passes=1)

    def test_covers_every_sample_once(self, loader, device):
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=4)
        assert len(result) == 32
        assert len(result.labels) == 32

    def test_records_the_pass_count(self, loader, device):
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=7)
        assert result.passes == 7

    def test_mean_probability_is_a_probability(self, loader, device):
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=8)
        assert result.mean_probability.min() >= 0.0
        assert result.mean_probability.max() <= 1.0

    def test_labels_survive_unpermuted(self, loader, device):
        # Labels are captured on the first pass only; they must still line up with
        # the predictions, which requires the loader to not shuffle.
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=3)
        expected = torch.cat([labels for _, labels in loader]).numpy()
        assert np.array_equal(result.labels, expected)

    def test_model_is_left_in_eval_mode(self, loader, device):
        # enable_mc_dropout puts the model in train mode. Leaving it there would
        # silently make a later evaluate() call stochastic.
        model = TinyDropoutNet()
        mc_dropout_predict(model, loader, device, passes=3)
        assert not model.training

    def test_epistemic_uncertainty_is_never_negative(self, loader, device):
        # Total minus aleatoric is non-negative in exact arithmetic; floating point
        # can cross zero and the clamp is what keeps it sane.
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=12)
        assert result.mutual_information.min() >= 0.0

    def test_epistemic_never_exceeds_total(self, loader, device):
        result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=12)
        assert np.all(result.mutual_information <= result.predictive_entropy + 1e-9)

    def test_more_dropout_gives_more_spread(self, loader, device):
        torch.manual_seed(1)
        light = mc_dropout_predict(TinyDropoutNet(p=0.05), loader, device, passes=25)
        torch.manual_seed(1)
        heavy = mc_dropout_predict(TinyDropoutNet(p=0.7), loader, device, passes=25)
        assert heavy.std_probability.mean() > light.std_probability.mean()


class TestTriage:
    def _result(self, probabilities, uncertainties, labels=None):
        probabilities = np.asarray(probabilities, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)
        return UncertaintyResult(
            mean_probability=probabilities,
            std_probability=np.zeros_like(probabilities),
            predictive_entropy=uncertainties,
            mutual_information=uncertainties,
            labels=np.zeros_like(probabilities, dtype=int)
            if labels is None
            else np.asarray(labels),
            passes=10,
        )

    def test_confident_predictions_bypass_the_queue(self):
        result = self._result([0.98, 0.01], [0.001, 0.001])
        masks = result.triage(affected_threshold=0.9, healthy_threshold=0.1)
        assert masks["act"][0]
        assert masks["ignore"][1]
        assert not masks["inspect"].any()

    def test_undecided_band_goes_to_inspection(self):
        result = self._result([0.5], [0.001])
        masks = result.triage(affected_threshold=0.9, healthy_threshold=0.1)
        assert masks["inspect"][0]

    def test_confident_but_uncertain_is_routed_to_a_human(self):
        # The case a threshold alone cannot find, and the reason this module pays
        # for itself: the model says 0.97 while disagreeing with itself internally.
        result = self._result([0.97, 0.97, 0.97, 0.97], [0.0, 0.0, 0.0, 0.9])
        masks = result.triage(
            affected_threshold=0.9, healthy_threshold=0.1, uncertainty_quantile=0.75
        )
        assert masks["inspect"][3]
        assert masks["act"][0]

    def test_masks_partition_the_samples(self):
        rng = np.random.default_rng(0)
        result = self._result(rng.random(100), rng.random(100))
        masks = result.triage(affected_threshold=0.8, healthy_threshold=0.2)
        total = (
            masks["act"].astype(int) + masks["ignore"].astype(int) + masks["inspect"].astype(int)
        )
        assert np.all(total == 1)

    def test_rejects_crossed_thresholds(self):
        result = self._result([0.5], [0.1])
        with pytest.raises(ValueError, match="must not exceed"):
            result.triage(affected_threshold=0.1, healthy_threshold=0.9)

    def test_rejects_an_out_of_range_quantile(self):
        result = self._result([0.5], [0.1])
        with pytest.raises(ValueError, match="quantile must be"):
            result.triage(affected_threshold=0.9, healthy_threshold=0.1, uncertainty_quantile=1.5)


class TestUncertaintySeparatesErrors:
    def _result(self, probabilities, uncertainties, labels):
        probabilities = np.asarray(probabilities, dtype=float)
        return UncertaintyResult(
            mean_probability=probabilities,
            std_probability=np.zeros_like(probabilities),
            predictive_entropy=np.asarray(uncertainties, dtype=float),
            mutual_information=np.asarray(uncertainties, dtype=float),
            labels=np.asarray(labels, dtype=int),
            passes=10,
        )

    def test_detects_the_useful_case(self):
        # Errors carry higher uncertainty, which is what makes the queue orderable.
        result = self._result(
            probabilities=[0.9, 0.9, 0.1, 0.1],
            uncertainties=[0.01, 0.5, 0.01, 0.5],
            labels=[1, 0, 0, 1],
        )
        summary = uncertainty_separates_errors(result)
        assert summary["separates"] is True
        assert summary["n_errors"] == 2

    def test_detects_the_useless_case(self):
        result = self._result(
            probabilities=[0.9, 0.9, 0.1, 0.1],
            uncertainties=[0.5, 0.01, 0.5, 0.01],
            labels=[1, 0, 0, 1],
        )
        assert uncertainty_separates_errors(result)["separates"] is False

    def test_a_perfect_model_reports_none_rather_than_crashing(self):
        # The situation this project keeps hitting: no errors to compare against.
        result = self._result([0.9, 0.1], [0.01, 0.01], [1, 0])
        summary = uncertainty_separates_errors(result)
        assert summary["separates"] is None
        assert summary["n_errors"] == 0

    def test_a_model_that_is_always_wrong_also_reports_none(self):
        result = self._result([0.9, 0.1], [0.01, 0.01], [0, 1])
        assert uncertainty_separates_errors(result)["separates"] is None


class TestRealBackbone:
    """MC Dropout on the actual architecture, not a toy net.

    timm's EfficientNet is the case d26341c was about: it reports zero
    `nn.Dropout` modules and carries `drop_rate` as an attribute instead, so the
    textbook recipe of toggling dropout modules silently samples a deterministic
    model.
    """

    @pytest.fixture
    def loader(self):
        torch.manual_seed(1)
        images = torch.randn(8, 3, 224, 224) * 0.5
        labels = torch.cat([torch.ones(4), torch.zeros(4)]).long()
        return DataLoader(TensorDataset(images, labels), batch_size=4)

    def test_timm_efficientnet_exposes_no_dropout_modules(self):
        # Pins the premise. If a timm release ever adds a real module here, the
        # functional path stops being the only one and this test says so.
        from citrus_scout.models.classifier import build_model

        model = build_model(num_classes=2, pretrained=False, dropout=0.3)
        assert sum(1 for m in model.modules() if isinstance(m, nn.Dropout)) == 0
        assert model.drop_rate == pytest.approx(0.3)

    def test_dropout_perturbs_the_logits(self):
        # The direct check, and the one that does not need trained weights: same
        # input twice in the MC state must give different logits.
        from citrus_scout.models.classifier import build_model, enable_mc_dropout

        torch.manual_seed(0)
        model = build_model(num_classes=2, pretrained=False, dropout=0.3)
        x = torch.randn(2, 3, 224, 224)

        enable_mc_dropout(model)
        with torch.no_grad():
            first, second = model(x), model(x)

        assert not torch.allclose(first, second), "dropout did not fire"

    def test_untrained_weights_give_a_tiny_spread(self):
        # Documents a trap. An untrained model's logits sit near zero, where
        # softmax is nearly flat, so real dropout noise still yields ~1e-5 of
        # probability spread. That is a model with no signal, not broken dropout:
        # diagnose it with the logit check above, never with the spread.
        from citrus_scout.models.classifier import build_model

        torch.manual_seed(0)
        model = build_model(num_classes=2, pretrained=False, dropout=0.3)
        images = torch.randn(4, 3, 224, 224)
        loader = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=4)

        result = mc_dropout_predict(model, loader, torch.device("cpu"), passes=10)
        assert result.std_probability.max() < 1e-2

    @pytest.mark.slow
    def test_pretrained_features_give_a_usable_spread(self, loader):
        # The deployed case. Downloads ImageNet weights, hence `slow`.
        from citrus_scout.models.classifier import build_model

        torch.manual_seed(0)
        model = build_model(num_classes=2, pretrained=True, dropout=0.3)

        result = mc_dropout_predict(model, loader, torch.device("cpu"), passes=20)

        assert result.std_probability.max() > 1e-2
        assert np.all(result.mutual_information >= 0.0)
        assert np.all(result.mutual_information <= result.predictive_entropy + 1e-9)
        assert not model.training
