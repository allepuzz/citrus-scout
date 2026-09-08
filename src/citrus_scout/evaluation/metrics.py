"""Metrics for classification under severe class imbalance.

The prevalence of affected trees in a well-managed commercial grove is low
(typically 2-5%). In that regime accuracy is useless: a model that always
predicts "healthy" is right 98% of the time while detecting nothing.

Worse, a model with high sensitivity and specificity can still be unusable in
practice, because what reaches the field technician is the set of predicted
positives — and most of them may be false. That is the number that decides
whether the product is viable: the **positive predictive value** at the real
field prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class OperatingPoint:
    """Classifier performance at one decision threshold."""

    threshold: float
    sensitivity: float
    """Recall / true positive rate: of the diseased trees, how many are detected."""
    specificity: float
    """Of the healthy trees, how many are correctly ruled out."""
    ppv: float
    """Positive predictive value at the evaluated prevalence: of the alerts raised, how many are real."""
    npv: float
    """Negative predictive value."""
    prevalence: float
    """Prevalence used to compute PPV and NPV."""

    def __str__(self) -> str:
        return (
            f"threshold={self.threshold:.3f} | sens={self.sensitivity:.1%} "
            f"spec={self.specificity:.1%} | PPV={self.ppv:.1%} "
            f"(prevalence {self.prevalence:.1%})"
        )


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value via Bayes' theorem.

    Projects performance measured on a balanced test set onto the real field
    prevalence, which is where the system will actually operate.

    Worked example: at 90% sensitivity, 90% specificity and 2% prevalence the PPV
    is 15.5% — roughly 6 out of every 7 alerts would be false. Raising specificity
    to 99% lifts the PPV to ~65%.

    >>> round(ppv_at_prevalence(0.90, 0.90, 0.02), 3)
    0.155
    >>> round(ppv_at_prevalence(0.90, 0.99, 0.02), 3)
    0.647
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")

    true_positives = sensitivity * prevalence
    false_positives = (1.0 - specificity) * (1.0 - prevalence)
    denominator = true_positives + false_positives

    if denominator == 0.0:
        return 0.0
    return true_positives / denominator


def npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Negative predictive value via Bayes' theorem."""
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")

    true_negatives = specificity * (1.0 - prevalence)
    false_negatives = (1.0 - sensitivity) * prevalence
    denominator = true_negatives + false_negatives

    if denominator == 0.0:
        return 0.0
    return true_negatives / denominator


def operating_point_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    prevalence: float,
) -> OperatingPoint:
    """Evaluate the classifier at one threshold, projecting PPV to the given prevalence."""
    y_pred = (y_score >= threshold).astype(int)

    positives = y_true == 1
    negatives = y_true == 0

    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())

    sensitivity = float((y_pred[positives] == 1).sum() / n_pos) if n_pos else 0.0
    specificity = float((y_pred[negatives] == 0).sum() / n_neg) if n_neg else 0.0

    return OperatingPoint(
        threshold=float(threshold),
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv_at_prevalence(sensitivity, specificity, prevalence),
        npv=npv_at_prevalence(sensitivity, specificity, prevalence),
        prevalence=prevalence,
    )


def threshold_for_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float,
) -> float:
    """Lowest threshold that reaches the target specificity.

    This is how the system's operating point is tuned. The cost of skipping a healthy
    tree is low, but flooding the technician with false positives makes them abandon
    the tool. Hence the operating point is set by specificity rather than by
    maximising F1.
    """
    if not 0.0 <= target_specificity <= 1.0:
        raise ValueError(f"specificity must be in [0, 1], got {target_specificity}")

    negative_scores = y_score[y_true == 0]
    if negative_scores.size == 0:
        raise ValueError("no negative samples available to compute specificity")

    # The threshold at the `target_specificity` quantile of the negative scores
    # leaves exactly that fraction of negatives below it.
    return float(np.quantile(negative_scores, target_specificity))


def classification_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    field_prevalence: float = 0.02,
    target_specificities: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, float | list[OperatingPoint]]:
    """Full report for an imbalanced binary classifier.

    Deliberately does **not** return accuracy: at low prevalence it is misleading and
    tends to justify models that detect nothing.

    Args:
        y_true: binary labels, 1 = affected.
        y_score: continuous score for the positive class (probability or logit).
        field_prevalence: expected real-world prevalence, used to project the PPV.
        target_specificities: specificities at which to report operating points.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")

    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"y_true must be binary (0/1), found {unique}")

    operating_points = [
        operating_point_at_threshold(
            y_true,
            y_score,
            threshold_for_specificity(y_true, y_score, spec),
            field_prevalence,
        )
        for spec in target_specificities
    ]

    # F1 at threshold 0.5, reported only for comparability with published results.
    f1_at_half = float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0))

    return {
        # PR-AUC is the headline metric: unlike ROC-AUC it is sensitive to imbalance.
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(unique) == 2 else float("nan"),
        "f1_at_0.5": f1_at_half,
        "test_prevalence": float(y_true.mean()),
        "field_prevalence": float(field_prevalence),
        "operating_points": operating_points,
    }
