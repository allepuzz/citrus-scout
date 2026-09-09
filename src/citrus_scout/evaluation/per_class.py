"""Per-class metrics and confusion analysis.

The binary report answers "does the model separate healthy from affected". It cannot
answer "what does it confuse", and on this data that is the more useful question.

A single PR-AUC hides structure that matters for where to spend effort. Aphids and
citrus leafminer are present in Murcia but show at millimetre scale; citrus canker
and HLB are absent from Spain and can only ever be pre-training material. A model
that scores well overall by nailing the absent classes while failing the present
ones is worthless here, and the aggregate number reads identically either way.

So this module reports per-class recall alongside each class's local relevance, and
weights the confusion pairs by whether confusing them would cost anything in a
Murcian grove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from citrus_scout.data.taxonomy import LocalRelevance, relevance_of


@dataclass(frozen=True)
class ClassMetrics:
    """How the model performs on one class."""

    label: str
    support: int
    """Samples of this class in the split."""

    recall: float
    """Of this class's samples, the fraction predicted correctly."""

    precision: float
    """Of the samples predicted as this class, the fraction that really are."""

    f1: float
    relevance: LocalRelevance
    """Whether the class occurs in Murcia. Governs how much its score matters."""

    @property
    def actionable(self) -> bool:
        """Whether a detection here would mean anything in a Murcian grove."""
        return self.relevance in (LocalRelevance.PRESENT, LocalRelevance.NONSPECIFIC)

    def __str__(self) -> str:
        return (
            f"{self.label}: recall={self.recall:.1%} precision={self.precision:.1%} "
            f"(n={self.support}, {self.relevance.value})"
        )


@dataclass(frozen=True)
class ConfusionPair:
    """One directed confusion: `true_label` predicted as `predicted_label`."""

    true_label: str
    predicted_label: str
    count: int
    rate: float
    """Fraction of `true_label` samples that went to `predicted_label`."""

    @property
    def costly(self) -> bool:
        """Whether this confusion would matter in the field.

        Mistaking one absent disease for another is free: neither occurs in Spain,
        so neither triggers an action. Mistaking a present disease for a healthy
        tree is a missed detection, and the reverse is a wasted visit. Only
        confusions touching a locally relevant class can cost anything.
        """
        return _locally_relevant(self.true_label) or _locally_relevant(self.predicted_label)

    def __str__(self) -> str:
        marker = " [costly]" if self.costly else ""
        return (
            f"{self.true_label} -> {self.predicted_label}: {self.count} ({self.rate:.1%}){marker}"
        )


def _locally_relevant(label: str) -> bool:
    """Whether a label names something that occurs in Murcia, healthy included."""
    return relevance_of(label) in (
        LocalRelevance.PRESENT,
        LocalRelevance.NONSPECIFIC,
        LocalRelevance.HEALTHY,
    )


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int | None = None,
) -> np.ndarray:
    """Counts of true class (rows) against predicted class (columns).

    Written out rather than taken from sklearn so that the class count is explicit:
    sklearn infers it from the labels present, which silently drops a class that
    never appears in either array and shifts every subsequent index.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1 if y_true.size else 0

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(true), int(predicted)] += 1
    return matrix


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> list[ClassMetrics]:
    """Recall, precision and F1 for every class, tagged with local relevance.

    A class with no samples in the split gets zeros rather than a nan, so the table
    renders and the missing support is visible in its own column.
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes=len(classes))

    results = []
    for index, label in enumerate(classes):
        support = int(matrix[index].sum())
        predicted_as = int(matrix[:, index].sum())
        correct = int(matrix[index, index])

        recall = correct / support if support else 0.0
        precision = correct / predicted_as if predicted_as else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append(
            ClassMetrics(
                label=label,
                support=support,
                recall=recall,
                precision=precision,
                f1=f1,
                relevance=relevance_of(label),
            )
        )
    return results


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    *,
    limit: int = 10,
    min_count: int = 1,
) -> list[ConfusionPair]:
    """The most frequent off-diagonal confusions, largest first.

    Ordered by rate rather than raw count, so a small class losing most of its
    samples outranks a large class losing a few. The small class is usually the
    more interesting failure.
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes=len(classes))

    pairs = []
    for true_index, true_label in enumerate(classes):
        support = matrix[true_index].sum()
        if support == 0:
            continue
        for predicted_index, predicted_label in enumerate(classes):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count < min_count:
                continue
            pairs.append(
                ConfusionPair(
                    true_label=true_label,
                    predicted_label=predicted_label,
                    count=count,
                    rate=count / support,
                )
            )

    pairs.sort(key=lambda p: (p.rate, p.count), reverse=True)
    return pairs[:limit]


def actionable_recall(metrics: list[ClassMetrics]) -> dict[str, float]:
    """Mean recall split by whether a class matters in Murcia.

    This is the comparison the aggregate metric cannot make. A model can post a
    strong overall score by learning the absent diseases, which are well
    represented in the public datasets, while failing the present ones that are
    the actual target. Seeing the two numbers side by side makes that visible.
    """
    actionable = [m for m in metrics if m.actionable and m.support > 0]
    other = [m for m in metrics if not m.actionable and m.support > 0]

    return {
        "actionable_mean_recall": float(np.mean([m.recall for m in actionable]))
        if actionable
        else float("nan"),
        "absent_mean_recall": float(np.mean([m.recall for m in other])) if other else float("nan"),
        "n_actionable_classes": len(actionable),
        "n_absent_classes": len(other),
    }
