"""Tests for imbalanced-classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from citrus_scout.evaluation.metrics import (
    classification_report,
    npv_at_prevalence,
    operating_point_at_threshold,
    ppv_at_prevalence,
    threshold_for_specificity,
)


class TestPPVAtPrevalence:
    def test_motivating_case(self):
        """90% sensitivity, 90% specificity, 2% prevalence -> 15.5% PPV.

        This is the example that drives the project's whole metrics approach:
        out of 1,000 trees, 20 are diseased, 18 are detected, and 98 false
        positives appear among the 980 healthy ones. PPV = 18 / (18 + 98) = 15.5%.
        """
        assert ppv_at_prevalence(0.90, 0.90, 0.02) == pytest.approx(0.1552, abs=1e-4)

    def test_higher_specificity_rescues_ppv(self):
        """Moving specificity from 90% to 99% lifts PPV from 15% to ~65%."""
        assert ppv_at_prevalence(0.90, 0.99, 0.02) == pytest.approx(0.6475, abs=1e-4)

    def test_perfect_classifier(self):
        assert ppv_at_prevalence(1.0, 1.0, 0.02) == 1.0

    def test_zero_prevalence_yields_no_true_positives(self):
        """At zero prevalence (the HLB case in Murcia) every positive is false."""
        assert ppv_at_prevalence(0.99, 0.99, 0.0) == 0.0

    def test_uninformative_classifier_returns_prevalence(self):
        """If sensitivity == 1 - specificity, the model carries no information."""
        prevalence = 0.02
        assert ppv_at_prevalence(0.5, 0.5, prevalence) == pytest.approx(prevalence)

    @pytest.mark.parametrize("prevalence", [-0.1, 1.5])
    def test_invalid_prevalence(self, prevalence):
        with pytest.raises(ValueError, match="prevalence"):
            ppv_at_prevalence(0.9, 0.9, prevalence)


class TestNPVAtPrevalence:
    def test_npv_is_high_at_low_prevalence(self):
        """For a rare disease, a negative call is almost certainly correct."""
        assert npv_at_prevalence(0.90, 0.90, 0.02) > 0.99

    def test_perfect_classifier(self):
        assert npv_at_prevalence(1.0, 1.0, 0.05) == 1.0


class TestThresholdForSpecificity:
    def test_reaches_requested_specificity(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(900), np.ones(100)])
        y_score = np.concatenate([rng.beta(2, 5, 900), rng.beta(5, 2, 100)])

        threshold = threshold_for_specificity(y_true, y_score, 0.95)
        op = operating_point_at_threshold(y_true, y_score, threshold, prevalence=0.02)

        assert op.specificity == pytest.approx(0.95, abs=0.02)

    def test_higher_specificity_needs_higher_threshold(self):
        rng = np.random.default_rng(0)
        y_true = np.concatenate([np.zeros(500), np.ones(500)])
        y_score = np.concatenate([rng.beta(2, 5, 500), rng.beta(5, 2, 500)])

        assert threshold_for_specificity(y_true, y_score, 0.99) > threshold_for_specificity(
            y_true, y_score, 0.90
        )

    def test_without_negatives(self):
        with pytest.raises(ValueError, match="negative"):
            threshold_for_specificity(np.ones(10), np.random.rand(10), 0.95)


class TestClassificationReport:
    @pytest.fixture
    def imbalanced_data(self):
        """1,000 samples at 2% prevalence with a reasonable classifier."""
        rng = np.random.default_rng(123)
        n_pos, n_neg = 20, 980
        y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        y_score = np.concatenate([rng.beta(6, 2, n_pos), rng.beta(2, 6, n_neg)])
        return y_true, y_score

    def test_does_not_report_accuracy(self, imbalanced_data):
        """Accuracy is excluded on purpose: it is misleading under imbalance."""
        report = classification_report(*imbalanced_data)
        assert not any("acc" in key.lower() for key in report)

    def test_includes_headline_metrics(self, imbalanced_data):
        report = classification_report(*imbalanced_data)
        assert "pr_auc" in report
        assert "operating_points" in report
        assert 0.0 <= report["pr_auc"] <= 1.0

    def test_one_operating_point_per_specificity(self, imbalanced_data):
        report = classification_report(*imbalanced_data, target_specificities=(0.90, 0.95, 0.99))
        assert len(report["operating_points"]) == 3

    def test_ppv_improves_with_specificity(self, imbalanced_data):
        report = classification_report(*imbalanced_data, target_specificities=(0.90, 0.99))
        low, high = report["operating_points"]
        assert high.ppv > low.ppv

    def test_records_both_prevalences(self, imbalanced_data):
        """Test-set and field prevalence differ; both must be recorded."""
        report = classification_report(*imbalanced_data, field_prevalence=0.03)
        assert report["test_prevalence"] == pytest.approx(0.02)
        assert report["field_prevalence"] == pytest.approx(0.03)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            classification_report(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))

    def test_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            classification_report(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
