"""Uncertainty quantification by MC Dropout.

The product does not just classify trees, it decides where to send a technician.
That makes "I do not know" a useful answer: a confident healthy call can be
skipped, a confident affected call goes straight to the inspection pass, and the
uncertain middle is where a human look is worth most.

Without this the system cannot express doubt. Every alert arrives with equal
weight, and the operating-point tables are the only tool for managing volume. The
baseline run showed what that costs: at 98.7% specificity, 1.7 alerts per genuine
case. Ranking the queue by uncertainty is what turns that number into a work plan
rather than a flat list.

MC Dropout approximates a Bayesian posterior cheaply: keep dropout active at
inference, run the same image several times, and read the spread of the
predictions. Wide spread means the model's answer depends on which units happened
to be dropped, which is as close to "this input is unlike my training data" as a
single deterministic network gets.

It is an approximation, not a calibrated posterior. Gal and Ghahramani (ICML 2016)
derive it as variational inference with a specific approximating family; the
spread it yields is informative for ranking, not a probability interval to quote
at a grower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from citrus_scout.models.classifier import enable_mc_dropout, has_active_dropout


@dataclass(frozen=True)
class UncertaintyResult:
    """Per-sample predictions with their uncertainty, plus the labels to score them."""

    mean_probability: np.ndarray
    """Posterior mean of the positive-class probability, averaged over passes."""

    std_probability: np.ndarray
    """Standard deviation across passes. The headline uncertainty signal."""

    predictive_entropy: np.ndarray
    """Entropy of the mean prediction: total uncertainty, aleatoric plus epistemic."""

    mutual_information: np.ndarray
    """Entropy of the mean minus mean entropy: the epistemic part alone.

    This is the component that more data could reduce, which makes it the better
    signal for deciding what to go and look at. Aleatoric uncertainty, a genuinely
    ambiguous image, does not get better by visiting the tree again.
    """

    labels: np.ndarray
    passes: int

    def __len__(self) -> int:
        return len(self.mean_probability)

    def triage(
        self,
        *,
        affected_threshold: float,
        healthy_threshold: float,
        uncertainty_quantile: float = 0.80,
    ) -> dict[str, np.ndarray]:
        """Split predictions into act / inspect / ignore by confidence.

        Returns boolean masks. `inspect` is the queue a technician works through:
        anything in the undecided band, plus anything the model called confidently
        while being internally unsure about it. That second group is the reason
        uncertainty earns its cost; a threshold alone cannot find it.
        """
        if not 0.0 <= uncertainty_quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {uncertainty_quantile}")
        if healthy_threshold > affected_threshold:
            raise ValueError(
                f"healthy_threshold ({healthy_threshold}) must not exceed "
                f"affected_threshold ({affected_threshold})"
            )

        cutoff = float(np.quantile(self.mutual_information, uncertainty_quantile))
        uncertain = self.mutual_information > cutoff

        confident_affected = (self.mean_probability >= affected_threshold) & ~uncertain
        confident_healthy = (self.mean_probability <= healthy_threshold) & ~uncertain

        return {
            "act": confident_affected,
            "ignore": confident_healthy,
            "inspect": ~(confident_affected | confident_healthy),
        }


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Binary entropy of each probability, in nats.

    Peaks at p=0.5 and falls to zero at both ends. Clipped away from the
    boundaries so that a confident p=0.0 or 1.0 gives 0 rather than a nan from
    log(0).
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    passes: int = 20,
    verify_dropout: bool = True,
) -> UncertaintyResult:
    """Run `passes` stochastic forward passes and summarise the spread.

    Raises when dropout is not actually active and `verify_dropout` is set. That
    check matters more than it looks: timm's EfficientNet applies dropout
    functionally from a `drop_rate` attribute rather than through an `nn.Dropout`
    module, so the usual recipe of toggling dropout modules finds nothing to
    toggle and silently samples a deterministic model. The result would be
    identical passes, zero variance, and a system reporting total confidence on
    every tree, including the ones it has no business being confident about.

    20 passes is the usual default and is enough to rank by; the standard error of
    the mean falls as 1/sqrt(passes), so doubling it buys about 40% tighter
    estimates for double the inference cost.
    """
    if passes < 2:
        raise ValueError(f"need at least 2 passes to measure spread, got {passes}")

    enable_mc_dropout(model)

    if verify_dropout and not has_active_dropout(model):
        raise RuntimeError(
            "dropout is not active, so every pass would be identical and the "
            "uncertainty would read as zero everywhere. Build the model with "
            "dropout > 0."
        )

    # Shape: (passes, n_samples). Collected per pass so the loader is consumed in
    # its fixed order each time and column i refers to the same image throughout.
    per_pass: list[np.ndarray] = []
    labels: np.ndarray | None = None

    for pass_index in range(passes):
        pass_probabilities: list[np.ndarray] = []
        pass_labels: list[np.ndarray] = []

        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            pass_probabilities.append(probabilities.cpu().numpy())
            if pass_index == 0:
                pass_labels.append(batch_labels.numpy())

        per_pass.append(np.concatenate(pass_probabilities))
        if pass_index == 0:
            labels = np.concatenate(pass_labels)

    model.eval()

    samples = np.stack(per_pass)  # (passes, n)
    mean = samples.mean(axis=0)

    # Total uncertainty: entropy of the averaged prediction.
    total = predictive_entropy(mean)
    # Aleatoric: average entropy of the individual predictions.
    aleatoric = predictive_entropy(samples).mean(axis=0)
    # Epistemic is what remains, and is never negative in exact arithmetic.
    epistemic = np.maximum(total - aleatoric, 0.0)

    assert labels is not None  # the loader yielded at least one batch
    return UncertaintyResult(
        mean_probability=mean,
        std_probability=samples.std(axis=0),
        predictive_entropy=total,
        mutual_information=epistemic,
        labels=labels,
        passes=passes,
    )


def uncertainty_separates_errors(result: UncertaintyResult, *, threshold: float = 0.5) -> dict:
    """Whether uncertainty is higher on the samples the model got wrong.

    This is the check that decides if the signal is worth routing on. If wrong
    predictions are not measurably more uncertain than right ones, the queue
    ordering carries no information and the extra inference cost buys nothing.
    """
    predictions = (result.mean_probability >= threshold).astype(int)
    wrong = predictions != result.labels

    n_wrong = int(wrong.sum())
    if n_wrong == 0 or n_wrong == len(result):
        # Nothing to compare against. Reported rather than raised: a perfect run
        # on an easy test set is exactly the situation this project keeps hitting.
        return {
            "n_errors": n_wrong,
            "mean_uncertainty_correct": float(result.mutual_information[~wrong].mean())
            if n_wrong < len(result)
            else float("nan"),
            "mean_uncertainty_wrong": float(result.mutual_information[wrong].mean())
            if n_wrong
            else float("nan"),
            "separates": None,
        }

    wrong_mi = float(result.mutual_information[wrong].mean())
    right_mi = float(result.mutual_information[~wrong].mean())

    return {
        "n_errors": n_wrong,
        "mean_uncertainty_correct": right_mi,
        "mean_uncertainty_wrong": wrong_mi,
        "ratio": wrong_mi / right_mi if right_mi > 0 else float("inf"),
        "separates": wrong_mi > right_mi,
    }
