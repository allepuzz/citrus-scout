"""Tests for probability calibration.

The property that matters most here is that temperature scaling cannot change the
ranking. If it could, calibrating would silently alter PR-AUC and the reported
"calibration improved things" would be hiding a changed model.
"""

from __future__ import annotations

import numpy as np
import pytest

from citrus_scout.evaluation.calibration import (
    apply_temperature,
    brier_score,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def overconfident_logits(n: int = 400, *, scale: float = 6.0, seed: int = 0):
    """Logits that rank well but state probabilities far too extreme.

    This is the shape the baseline run produced: near-perfect separation, with all
    the probability mass pushed against 0 and 1.

    At the default scale some probabilities round to exactly 1.0 in float32, which
    is realistic but destroys the information needed to compare orderings. Tests
    about monotonicity pass a smaller `scale`.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Correct side of the boundary most of the time, but hugely over-scaled.
    signal = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.5, size=n)
    logits = np.stack([-signal * scale, signal * scale], axis=1)
    return logits, labels


class TestExpectedCalibrationError:
    def test_perfect_calibration_scores_zero(self):
        # Half the samples at p=0.0 all negative, half at p=1.0 all positive.
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)

    def test_confident_and_wrong_scores_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(1.0, abs=1e-9)

    def test_a_well_calibrated_coin_scores_near_zero(self):
        # Every prediction is 0.5 and exactly half are positive.
        y_true = np.array([0, 1] * 50)
        y_prob = np.full(100, 0.5)
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_is_penalised(self):
        # 70% of these are positive, so 0.7 is the calibrated answer. Claiming
        # 0.99 is the same ranking (all ties) but overstates the confidence.
        y_true = np.array([1] * 70 + [0] * 30)
        honest = np.full(100, 0.7)
        overconfident = np.full(100, 0.99)
        assert expected_calibration_error(y_true, overconfident) > expected_calibration_error(
            y_true, honest
        )

    def test_probability_of_exactly_one_is_binned(self):
        # A p=1.0 must land in the last bin, not fall outside the range.
        assert expected_calibration_error(np.array([1]), np.array([1.0])) == pytest.approx(0.0)

    def test_empty_input_is_zero_not_an_error(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            expected_calibration_error(np.array([0, 1]), np.array([0.5]))

    def test_rejects_zero_bins(self):
        with pytest.raises(ValueError, match="n_bins must be positive"):
            expected_calibration_error(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=0)

    def test_empty_bins_do_not_count_as_calibrated(self):
        # All mass in one bin, badly calibrated. Empty bins must not dilute it.
        y_true = np.zeros(50)
        y_prob = np.full(50, 0.95)
        assert expected_calibration_error(y_true, y_prob, n_bins=50) == pytest.approx(
            0.95, abs=0.01
        )


class TestBrierScore:
    def test_perfect_prediction_scores_zero(self):
        assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_worst_prediction_scores_one(self):
        assert brier_score(np.array([0, 1]), np.array([1.0, 0.0])) == pytest.approx(1.0)

    def test_hedging_scores_a_quarter(self):
        assert brier_score(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.25)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            brier_score(np.array([0, 1]), np.array([0.5]))


class TestFitTemperature:
    def test_already_calibrated_logits_keep_t_near_one(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, size=500)
        signal = np.where(labels == 1, 0.8, -0.8) + rng.normal(0, 1.0, size=500)
        logits = np.stack([-signal, signal], axis=1)
        assert fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.35)

    def test_overconfident_logits_give_t_above_one(self):
        # The correction for confidence that outruns accuracy is to divide by T > 1.
        logits, labels = overconfident_logits()
        assert fit_temperature(logits, labels) > 1.2

    def test_temperature_is_always_positive(self):
        # Optimising log(T) is what guarantees this; a T <= 0 would invert the ranking.
        logits, labels = overconfident_logits(seed=7)
        assert fit_temperature(logits, labels) > 0.0

    def test_rejects_one_dimensional_logits(self):
        with pytest.raises(ValueError, match="logits must be 2D"):
            fit_temperature(np.array([0.1, 0.9]), np.array([0, 1]))

    def test_rejects_row_mismatch(self):
        with pytest.raises(ValueError, match="row mismatch"):
            fit_temperature(np.zeros((5, 2)), np.zeros(3))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty set"):
            fit_temperature(np.zeros((0, 2)), np.zeros(0))


class TestApplyTemperature:
    def test_temperature_of_one_is_the_identity(self):
        logits = np.array([[-2.0, 2.0], [1.0, -1.0]])
        plain = apply_temperature(logits, 1.0)
        assert plain == pytest.approx([0.982014, 0.119203], abs=1e-5)

    def test_high_temperature_moves_probabilities_towards_half(self):
        logits = np.array([[-5.0, 5.0]])
        assert apply_temperature(logits, 10.0)[0] < apply_temperature(logits, 1.0)[0]

    def test_low_temperature_sharpens(self):
        logits = np.array([[-1.0, 1.0]])
        assert apply_temperature(logits, 0.5)[0] > apply_temperature(logits, 1.0)[0]

    def test_ranking_is_preserved(self):
        # The load-bearing property: scaling by a positive scalar is monotonic, so
        # PR-AUC and ROC-AUC cannot move. If this breaks, calibration is silently
        # changing the model's discrimination.
        logits, _ = overconfident_logits(seed=3, scale=1.5)
        raw = apply_temperature(logits, 1.0)
        scaled = apply_temperature(logits, 2.7)
        assert np.array_equal(raw.argsort().argsort(), scaled.argsort().argsort())

    def test_ranking_preserved_for_sharpening_too(self):
        # T < 1 multiplies the logits, so the scale has to stay small enough that
        # the sharpened probabilities do not round to 0 and 1 in float32. Past
        # that point the ordering is lost to the float format, not to the maths.
        logits, _ = overconfident_logits(seed=4, scale=0.8)
        raw = apply_temperature(logits, 1.0)
        scaled = apply_temperature(logits, 0.5)
        assert np.array_equal(raw.argsort().argsort(), scaled.argsort().argsort())

    def test_saturation_is_a_float_limit_not_a_maths_error(self):
        # Documents the boundary the two tests above avoid: at extreme scales
        # probabilities collapse onto exactly 0.0 and 1.0 and ties appear. Worth
        # pinning, because it is the regime the real model operates in, and it is
        # why ranking is compared on logits rather than probabilities elsewhere.
        logits, _ = overconfident_logits(seed=4, scale=8.0)
        saturated = apply_temperature(logits, 1.0)
        assert np.any(saturated == 1.0)

    def test_pr_auc_is_unchanged_by_scaling(self):
        # The consequence that actually matters, stated on the metric itself.
        from sklearn.metrics import average_precision_score

        logits, labels = overconfident_logits(seed=23, scale=1.5)
        raw = average_precision_score(labels, apply_temperature(logits, 1.0))
        scaled = average_precision_score(labels, apply_temperature(logits, 3.3))
        assert raw == pytest.approx(scaled, abs=1e-9)

    def test_output_is_a_probability(self):
        logits, _ = overconfident_logits(seed=5)
        probabilities = apply_temperature(logits, 1.8)
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_rejects_zero_temperature(self):
        with pytest.raises(ValueError, match="must be positive"):
            apply_temperature(np.array([[0.0, 1.0]]), 0.0)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError, match="must be positive"):
            apply_temperature(np.array([[0.0, 1.0]]), -1.0)


class TestCalibrate:
    def test_overconfidence_is_detected(self):
        logits, labels = overconfident_logits()
        _, report = calibrate(logits, labels)
        assert report.temperature > 1.0

    def test_ece_improves_when_miscalibration_is_the_dominant_error(self):
        # ECE is a binned statistic, and on near-perfectly separated data it is
        # already tiny, small enough that binning noise can move it either way.
        # Overlapping classes make the miscalibration real and measurable.
        rng = np.random.default_rng(29)
        labels = rng.integers(0, 2, size=2000)
        signal = np.where(labels == 1, 0.45, -0.45) + rng.normal(0, 1.0, size=2000)
        logits = np.stack([-signal * 4.0, signal * 4.0], axis=1)

        _, report = calibrate(logits, labels)
        assert report.temperature > 1.0
        assert report.improved()
        assert report.ece_after < report.ece_before

    def test_nll_does_not_get_worse(self):
        # T=1 is in the search space, so the optimum can never be worse than raw.
        logits, labels = overconfident_logits(seed=11)
        _, report = calibrate(logits, labels)
        assert report.nll_after <= report.nll_before + 1e-6

    def test_report_renders_a_direction(self):
        logits, labels = overconfident_logits()
        _, report = calibrate(logits, labels)
        assert "overconfident" in str(report)

    def test_returned_temperature_matches_the_report(self):
        logits, labels = overconfident_logits(seed=13)
        temperature, report = calibrate(logits, labels)
        assert temperature == pytest.approx(report.temperature)


class TestReliabilityCurve:
    def test_skips_empty_bins(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.05, 0.95])
        confidence, _frequency, counts = reliability_curve(y_true, y_prob, n_bins=10)
        assert len(confidence) == 2
        assert counts.sum() == 2

    def test_perfect_model_lies_on_the_diagonal(self):
        y_true = np.array([0] * 20 + [1] * 20)
        y_prob = np.array([0.0] * 20 + [1.0] * 20)
        confidence, frequency, _ = reliability_curve(y_true, y_prob, n_bins=10)
        assert confidence == pytest.approx(frequency, abs=1e-9)

    def test_counts_sum_to_the_sample_size(self):
        logits, labels = overconfident_logits(n=200, seed=17)
        probabilities = apply_temperature(logits, 1.0)
        _, _, counts = reliability_curve(labels, probabilities)
        assert counts.sum() == 200
