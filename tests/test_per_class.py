"""Tests for per-class metrics and confusion analysis.

The point of this module is that an aggregate score cannot distinguish a model that
learned the diseases present in Murcia from one that learned the absent ones. The
tests below state that distinction directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from citrus_scout.data.taxonomy import LocalRelevance
from citrus_scout.evaluation.per_class import (
    actionable_recall,
    confusion_matrix,
    per_class_metrics,
    top_confusions,
)

# Real labels from the project taxonomy, chosen to span the relevance tags.
PRESENT = "gummosis"  # Phytophthora, present in Murcia
PRESENT_2 = "aphids"
ABSENT = "citrus canker"  # quarantine pathogen, absent from Spain
ABSENT_2 = "greening"  # HLB, absent
HEALTHY = "healthy"

# Spans every relevance tag: one healthy, two present, two absent.
ALL_CLASSES = [HEALTHY, PRESENT, PRESENT_2, ABSENT, ABSENT_2]


class TestConfusionMatrix:
    def test_perfect_predictions_are_diagonal(self):
        matrix = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert np.array_equal(matrix, np.eye(3, dtype=int))

    def test_counts_land_in_the_right_cell(self):
        # One sample of true class 0 predicted as class 1.
        matrix = confusion_matrix(np.array([0]), np.array([1]), n_classes=2)
        assert matrix[0, 1] == 1
        assert matrix.sum() == 1

    def test_rows_are_true_and_columns_predicted(self):
        # Asymmetric case, so a transposed implementation fails here.
        matrix = confusion_matrix(np.array([0, 0, 1]), np.array([1, 1, 1]), n_classes=2)
        assert matrix[0, 1] == 2
        assert matrix[1, 0] == 0

    def test_total_equals_sample_count(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=200)
        y_pred = rng.integers(0, 4, size=200)
        assert confusion_matrix(y_true, y_pred, n_classes=4).sum() == 200

    def test_explicit_class_count_keeps_absent_classes(self):
        # The reason this is not sklearn's: class 2 appears nowhere, but the matrix
        # must still be 3x3 or every later index shifts.
        matrix = confusion_matrix(np.array([0, 1]), np.array([0, 1]), n_classes=3)
        assert matrix.shape == (3, 3)
        assert matrix[2].sum() == 0

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            confusion_matrix(np.array([0, 1]), np.array([0]))


class TestPerClassMetrics:
    def test_perfect_model_scores_one_everywhere(self):
        metrics = per_class_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), [HEALTHY, PRESENT]
        )
        assert all(m.recall == 1.0 and m.precision == 1.0 for m in metrics)

    def test_recall_and_precision_differ_as_expected(self):
        # Class 1 has 2 samples, 1 found (recall 0.5). Two predictions of class 1,
        # one correct (precision 0.5).
        metrics = per_class_metrics(
            np.array([0, 1, 1, 0]), np.array([1, 1, 0, 0]), [HEALTHY, PRESENT]
        )
        affected = next(m for m in metrics if m.label == PRESENT)
        assert affected.recall == pytest.approx(0.5)
        assert affected.precision == pytest.approx(0.5)

    def test_support_counts_true_samples(self):
        metrics = per_class_metrics(
            np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]), [HEALTHY, PRESENT]
        )
        assert next(m for m in metrics if m.label == HEALTHY).support == 3
        assert next(m for m in metrics if m.label == PRESENT).support == 1

    def test_missing_class_gets_zeros_not_nan(self):
        # A class absent from the split must not poison the table with nan.
        metrics = per_class_metrics(np.array([0, 0]), np.array([0, 0]), [HEALTHY, PRESENT])
        missing = next(m for m in metrics if m.label == PRESENT)
        assert missing.support == 0
        assert missing.recall == 0.0
        assert not np.isnan(missing.f1)

    def test_f1_is_the_harmonic_mean(self):
        metrics = per_class_metrics(
            np.array([0, 1, 1, 1]), np.array([1, 1, 1, 0]), [HEALTHY, PRESENT]
        )
        affected = next(m for m in metrics if m.label == PRESENT)
        expected = 2 * affected.precision * affected.recall / (affected.precision + affected.recall)
        assert affected.f1 == pytest.approx(expected)

    def test_relevance_comes_from_the_taxonomy(self):
        metrics = per_class_metrics(
            np.array([0, 1, 2]), np.array([0, 1, 2]), [HEALTHY, PRESENT, ABSENT]
        )
        by_label = {m.label: m for m in metrics}
        assert by_label[HEALTHY].relevance is LocalRelevance.HEALTHY
        assert by_label[PRESENT].relevance is LocalRelevance.PRESENT
        assert by_label[ABSENT].relevance is LocalRelevance.ABSENT

    def test_absent_disease_is_not_actionable(self):
        # The distinction the whole module exists for: canker cannot be acted on in
        # Murcia no matter how well it is detected.
        metrics = per_class_metrics(np.array([0]), np.array([0]), [ABSENT])
        assert not metrics[0].actionable

    def test_present_disease_is_actionable(self):
        metrics = per_class_metrics(np.array([0]), np.array([0]), [PRESENT])
        assert metrics[0].actionable

    def test_binary_affected_label_counts_as_actionable(self):
        # "affected" is not in the taxonomy and falls through to NONSPECIFIC, which
        # is treated as actionable. Correct for the binary task, and pinned here so
        # a change to the default is noticed.
        metrics = per_class_metrics(np.array([0, 1]), np.array([0, 1]), [HEALTHY, "affected"])
        assert next(m for m in metrics if m.label == "affected").actionable


class TestTopConfusions:
    def test_a_perfect_model_has_no_confusions(self):
        assert top_confusions(np.array([0, 1]), np.array([0, 1]), [HEALTHY, PRESENT]) == []

    def test_finds_the_confusion(self):
        pairs = top_confusions(np.array([0, 0]), np.array([1, 1]), [HEALTHY, PRESENT])
        assert len(pairs) == 1
        assert pairs[0].true_label == HEALTHY
        assert pairs[0].predicted_label == PRESENT
        assert pairs[0].count == 2
        assert pairs[0].rate == pytest.approx(1.0)

    def test_ordered_by_rate_so_small_classes_surface(self):
        # Class 1 loses 1 of 1 (rate 1.0); class 0 loses 2 of 100 (rate 0.02). The
        # small total loss is the more interesting failure and must rank first.
        y_true = np.array([0] * 100 + [1])
        y_pred = np.array([0] * 98 + [1, 1] + [0])
        pairs = top_confusions(y_true, y_pred, [HEALTHY, PRESENT])
        assert pairs[0].true_label == PRESENT
        assert pairs[0].rate == pytest.approx(1.0)

    def test_respects_the_limit(self):
        rng = np.random.default_rng(1)
        classes = [HEALTHY, PRESENT, PRESENT_2, ABSENT, ABSENT_2]
        y_true = rng.integers(0, 5, size=300)
        y_pred = rng.integers(0, 5, size=300)
        assert len(top_confusions(y_true, y_pred, classes, limit=3)) == 3

    def test_min_count_filters_noise(self):
        y_true = np.array([0] * 50 + [1])
        y_pred = np.array([0] * 49 + [1] + [1])
        assert top_confusions(y_true, y_pred, [HEALTHY, PRESENT], min_count=2) == []

    def test_confusion_between_two_absent_diseases_is_free(self):
        # Neither occurs in Spain, so mixing them up costs nothing in the field.
        pairs = top_confusions(np.array([0, 0]), np.array([1, 1]), [ABSENT, ABSENT_2])
        assert not pairs[0].costly

    def test_missing_a_present_disease_is_costly(self):
        pairs = top_confusions(np.array([0]), np.array([1]), [PRESENT, ABSENT])
        assert pairs[0].costly

    def test_calling_a_healthy_tree_diseased_is_costly(self):
        # A wasted visit, which is the cost the operating point is tuned against.
        pairs = top_confusions(np.array([0]), np.array([1]), [HEALTHY, ABSENT])
        assert pairs[0].costly

    def test_skips_classes_with_no_samples(self):
        pairs = top_confusions(np.array([0, 0]), np.array([0, 0]), [HEALTHY, PRESENT, ABSENT])
        assert pairs == []


class TestActionableRecall:
    def test_separates_actionable_from_absent(self):
        # The headline case: perfect on the absent disease, useless on the present
        # one. An aggregate score would look respectable; this must not.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # all predicted as the absent class
        metrics = per_class_metrics(y_true, y_pred, [PRESENT, ABSENT])
        summary = actionable_recall(metrics)

        assert summary["actionable_mean_recall"] == pytest.approx(0.0)
        assert summary["absent_mean_recall"] == pytest.approx(1.0)

    def test_counts_the_classes_in_each_group(self):
        metrics = per_class_metrics(
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),
            [PRESENT, PRESENT_2, ABSENT, ABSENT_2],
        )
        summary = actionable_recall(metrics)
        assert summary["n_actionable_classes"] == 2
        assert summary["n_absent_classes"] == 2

    def test_healthy_counts_as_neither(self):
        # Healthy is its own relevance tag: not a disease to act on, not absent.
        metrics = per_class_metrics(np.array([0, 1]), np.array([0, 1]), [HEALTHY, PRESENT])
        summary = actionable_recall(metrics)
        assert summary["n_actionable_classes"] == 1

    def test_empty_group_gives_nan_not_a_crash(self):
        metrics = per_class_metrics(np.array([0]), np.array([0]), [PRESENT])
        summary = actionable_recall(metrics)
        assert np.isnan(summary["absent_mean_recall"])
        assert summary["actionable_mean_recall"] == pytest.approx(1.0)

    def test_classes_with_no_support_are_excluded(self):
        # A class absent from the split must not drag the mean down with its zero.
        metrics = per_class_metrics(np.array([0, 0]), np.array([0, 0]), [PRESENT, PRESENT_2])
        summary = actionable_recall(metrics)
        assert summary["n_actionable_classes"] == 1
        assert summary["actionable_mean_recall"] == pytest.approx(1.0)


class TestAggregateMetricBlindSpot:
    """The scenario this whole module exists for.

    Two models with identical accuracy, opposite field value. An aggregate number
    reads the same for both; the per-class view separates them.
    """

    @staticmethod
    def _truth():
        return np.repeat(np.arange(5), 100)

    def _good_locally(self):
        # Confuses the two absent diseases with each other, nails the present ones.
        pred = self._truth().copy()
        pred[300:370] = 4
        pred[400:470] = 3
        return pred

    def _good_on_absent(self):
        # The mirror image: fails the present diseases, nails the absent ones.
        pred = self._truth().copy()
        pred[100:170] = 2
        pred[200:270] = 1
        return pred

    def test_the_two_models_have_identical_accuracy(self):
        truth = self._truth()
        assert (self._good_locally() == truth).mean() == pytest.approx(
            (self._good_on_absent() == truth).mean()
        )

    def test_per_class_view_separates_them(self):
        truth = self._truth()

        useful = actionable_recall(per_class_metrics(truth, self._good_locally(), ALL_CLASSES))
        misleading = actionable_recall(
            per_class_metrics(truth, self._good_on_absent(), ALL_CLASSES)
        )

        # The model that works on Murcia scores higher where it counts.
        assert useful["actionable_mean_recall"] > useful["absent_mean_recall"]
        # The other one has the relationship inverted, which is the warning sign.
        assert misleading["absent_mean_recall"] > misleading["actionable_mean_recall"]

    def test_cost_tagging_matches_the_verdict(self):
        truth = self._truth()

        free = top_confusions(truth, self._good_locally(), ALL_CLASSES)
        costly = top_confusions(truth, self._good_on_absent(), ALL_CLASSES)

        # Confusing canker with HLB costs nothing here; confusing gummosis with
        # aphids sends a technician to the wrong tree with the wrong treatment.
        assert not any(pair.costly for pair in free)
        assert all(pair.costly for pair in costly)
