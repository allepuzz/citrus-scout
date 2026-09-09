"""Probability calibration.

A classifier can rank perfectly and still report probabilities that mean nothing.
The baseline run is a clear case: it reached PR-AUC 1.0, yet the thresholds needed
to hit 100% sensitivity were 0.009 to 0.046. Scores that extreme are not
probabilities, they are confidences piled against the ends of the interval.

That matters here because the operating point is chosen *on* those numbers. The
project tunes for specificity and projects PPV to field prevalence, and both
arguments assume a score of 0.8 means roughly an 80% chance of being affected.
When the model is overconfident the chosen threshold does not deliver the
specificity it promised.

Temperature scaling is the fix used here: divide the logits by a single learned
scalar T before the softmax. With T > 1 probabilities move towards 0.5, with
T < 1 away from it. One parameter, fitted on validation data, and because it is a
monotonic transform of the logits it cannot change the ranking, so PR-AUC and
ROC-AUC are untouched. It improves how the numbers read without touching what the
model knows.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CalibrationReport:
    """How well probabilities match observed frequencies, before and after scaling."""

    temperature: float
    """Fitted scalar. Above 1 means the model was overconfident."""

    ece_before: float
    """Expected calibration error of the raw probabilities."""

    ece_after: float
    """Expected calibration error after temperature scaling."""

    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float

    def improved(self) -> bool:
        """Whether scaling actually helped on the data it was measured on."""
        return self.ece_after < self.ece_before

    def __str__(self) -> str:
        direction = "overconfident" if self.temperature > 1.0 else "underconfident"
        return (
            f"T={self.temperature:.3f} ({direction}) | "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f} | "
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f}"
        )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected calibration error: mean gap between confidence and accuracy.

    Predictions are binned by confidence, and each bin contributes the absolute
    difference between its mean predicted probability and the fraction actually
    positive, weighted by how many samples it holds. Zero means every stated
    probability matches its observed frequency.

    Empty bins are skipped rather than counted as perfect, which would otherwise
    let a model with all its mass in one bin look well calibrated.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if y_true.size == 0:
        return 0.0

    # Right-closed bins so that a probability of exactly 1.0 lands in the last one
    # rather than falling outside the range.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    error = 0.0
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += (count / y_true.size) * abs(confidence - accuracy)

    return float(error)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between probabilities and outcomes.

    Unlike ECE this is a proper scoring rule: it cannot be gamed by predicting the
    base rate everywhere, so it rewards being both calibrated and discriminative.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points for a reliability diagram.

    Returns mean confidence, observed frequency, and sample count per non-empty
    bin. A perfectly calibrated model puts every point on the diagonal; points
    below it are overconfident.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    confidences, frequencies, counts = [], [], []
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidences.append(float(y_prob[mask].mean()))
        frequencies.append(float(y_true[mask].mean()))
        counts.append(count)

    return np.array(confidences), np.array(frequencies), np.array(counts, dtype=int)


def fit_temperature(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit the temperature that minimises NLL on held-out data.

    Must be fitted on validation logits, never on test: the point is to correct
    confidence on data the model did not train on, and fitting on test would
    report a calibration that does not exist anywhere else.

    Optimises log(T) rather than T directly, which keeps the temperature positive
    without needing a constrained optimiser. A non-positive T would flip or destroy
    the ranking.
    """
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    if logits_t.ndim != 2:
        raise ValueError(f"logits must be 2D (n, classes), got shape {tuple(logits_t.shape)}")
    if logits_t.shape[0] != labels_t.shape[0]:
        raise ValueError(f"row mismatch: {logits_t.shape[0]} logits vs {labels_t.shape[0]} labels")
    if logits_t.shape[0] == 0:
        raise ValueError("cannot fit a temperature on an empty set")

    log_t = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1, the identity
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits_t / torch.exp(log_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def apply_temperature(
    logits: np.ndarray | torch.Tensor,
    temperature: float,
) -> np.ndarray:
    """Positive-class probabilities after scaling logits by `temperature`."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    probabilities = torch.softmax(logits_t / temperature, dim=1)
    return probabilities[:, 1].numpy()


def calibrate(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> tuple[float, CalibrationReport]:
    """Fit a temperature and measure what it changed.

    Both the fit and the measurement use the same validation set, so the reported
    improvement is optimistic by construction. It answers "did scaling help at
    all", not "by how much will it help on test"; for that, apply the returned
    temperature to the test logits and measure there.
    """
    temperature = fit_temperature(val_logits, val_labels)

    before = apply_temperature(val_logits, 1.0)
    after = apply_temperature(val_logits, temperature)

    labels = np.asarray(val_labels, dtype=float).ravel()

    report = CalibrationReport(
        temperature=temperature,
        ece_before=expected_calibration_error(labels, before, n_bins=n_bins),
        ece_after=expected_calibration_error(labels, after, n_bins=n_bins),
        brier_before=brier_score(labels, before),
        brier_after=brier_score(labels, after),
        nll_before=_nll(val_logits, val_labels, 1.0),
        nll_after=_nll(val_logits, val_labels, temperature),
    )
    return temperature, report


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Negative log likelihood at a given temperature, the quantity the fit minimises."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    loss = nn.functional.cross_entropy(logits_t / temperature, labels_t)
    return float(loss.item())
