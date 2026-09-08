"""Métricas para clasificación con fuerte desbalanceo de clases.

La prevalencia de árboles afectados en un huerto comercial bien gestionado es baja
(típicamente 2-5 %). En ese régimen la *accuracy* es inútil: un modelo que predice
siempre "sano" acierta el 98 % de las veces y no detecta nada.

Peor aún, un modelo con sensibilidad y especificidad altas puede seguir siendo
inservible en la práctica, porque lo que le llega al técnico son los positivos
predichos, y la mayoría pueden ser falsos. Ese es el número que decide si el
producto sirve: el **valor predictivo positivo** a la prevalencia real de campo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class OperatingPoint:
    """Rendimiento del clasificador en un umbral de decisión concreto."""

    threshold: float
    sensitivity: float
    """Recall / tasa de verdaderos positivos: de los enfermos, cuántos detecto."""
    specificity: float
    """De los sanos, cuántos descarto correctamente."""
    ppv: float
    """Valor predictivo positivo a la prevalencia evaluada: de mis alertas, cuántas son reales."""
    npv: float
    """Valor predictivo negativo."""
    prevalence: float
    """Prevalencia usada para calcular VPP y VPN."""

    def __str__(self) -> str:
        return (
            f"umbral={self.threshold:.3f} | sens={self.sensitivity:.1%} "
            f"espec={self.specificity:.1%} | VPP={self.ppv:.1%} "
            f"(prevalencia {self.prevalence:.1%})"
        )


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Valor predictivo positivo mediante el teorema de Bayes.

    Permite proyectar el rendimiento medido en un conjunto de test balanceado a la
    prevalencia real de campo, que es donde el sistema se va a usar.

    Ejemplo del informe base: con sensibilidad 90 %, especificidad 90 % y prevalencia
    2 %, el VPP es del 15,5 % — alrededor de 6 de cada 7 alertas serían falsas.
    Subiendo la especificidad al 99 %, el VPP sube al ~65 %.

    >>> round(ppv_at_prevalence(0.90, 0.90, 0.02), 3)
    0.155
    >>> round(ppv_at_prevalence(0.90, 0.99, 0.02), 3)
    0.647
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"La prevalencia debe estar en [0, 1], recibido {prevalence}")

    true_positives = sensitivity * prevalence
    false_positives = (1.0 - specificity) * (1.0 - prevalence)
    denominator = true_positives + false_positives

    if denominator == 0.0:
        return 0.0
    return true_positives / denominator


def npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Valor predictivo negativo mediante el teorema de Bayes."""
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"La prevalencia debe estar en [0, 1], recibido {prevalence}")

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
    """Evalúa el clasificador en un umbral, proyectando el VPP a la prevalencia dada."""
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
    """Umbral más bajo que alcanza la especificidad objetivo.

    Este es el ajuste operativo del sistema. El coste de no inspeccionar un árbol sano
    es bajo, pero saturar al técnico de falsos positivos hace que deje de usar la
    herramienta. Por eso el punto de operación se fija por especificidad, no por
    maximizar F1.
    """
    if not 0.0 <= target_specificity <= 1.0:
        raise ValueError(f"La especificidad debe estar en [0, 1], recibido {target_specificity}")

    negative_scores = y_score[y_true == 0]
    if negative_scores.size == 0:
        raise ValueError("No hay muestras negativas para calcular la especificidad")

    # El umbral en el percentil `target_specificity` de las puntuaciones negativas
    # deja exactamente esa fracción de negativos por debajo.
    return float(np.quantile(negative_scores, target_specificity))


def classification_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    field_prevalence: float = 0.02,
    target_specificities: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, float | list[OperatingPoint]]:
    """Informe completo para un clasificador binario desbalanceado.

    Deliberadamente **no** devuelve accuracy: con prevalencias bajas es engañosa y
    tiende a justificar modelos que no detectan nada.

    Args:
        y_true: etiquetas binarias, 1 = afectado.
        y_score: puntuación continua de la clase positiva (probabilidad o logit).
        field_prevalence: prevalencia real esperada en campo, para proyectar el VPP.
        target_specificities: especificidades a las que reportar puntos de operación.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(f"Dimensiones incompatibles: {y_true.shape} vs {y_score.shape}")

    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"y_true debe ser binario (0/1), encontrado {unique}")

    operating_points = [
        operating_point_at_threshold(
            y_true,
            y_score,
            threshold_for_specificity(y_true, y_score, spec),
            field_prevalence,
        )
        for spec in target_specificities
    ]

    # F1 en el umbral 0.5, solo como referencia comparable con la literatura.
    f1_at_half = float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0))

    return {
        # PR-AUC es la métrica principal: insensible al desbalanceo, a diferencia de ROC-AUC.
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(unique) == 2 else float("nan"),
        "f1_at_0.5": f1_at_half,
        "test_prevalence": float(y_true.mean()),
        "field_prevalence": float(field_prevalence),
        "operating_points": operating_points,
    }
