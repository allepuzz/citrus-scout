"""Tests de las métricas de clasificación desbalanceada."""

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
    def test_caso_del_informe(self):
        """Sensibilidad 90 %, especificidad 90 %, prevalencia 2 % -> VPP 15,5 %.

        Es el ejemplo que motiva todo el enfoque de métricas del proyecto:
        de 1.000 árboles hay 20 enfermos, se detectan 18, y aparecen 98 falsos
        positivos de los 980 sanos. VPP = 18 / (18 + 98) = 15,5 %.
        """
        assert ppv_at_prevalence(0.90, 0.90, 0.02) == pytest.approx(0.1552, abs=1e-4)

    def test_subir_especificidad_rescata_el_vpp(self):
        """Pasar la especificidad del 90 % al 99 % lleva el VPP del 15 % al ~65 %."""
        assert ppv_at_prevalence(0.90, 0.99, 0.02) == pytest.approx(0.6475, abs=1e-4)

    def test_clasificador_perfecto(self):
        assert ppv_at_prevalence(1.0, 1.0, 0.02) == 1.0

    def test_prevalencia_cero_no_da_positivos_reales(self):
        """Con prevalencia 0 (caso del HLB en Murcia) todo positivo es falso."""
        assert ppv_at_prevalence(0.99, 0.99, 0.0) == 0.0

    def test_clasificador_inutil_devuelve_la_prevalencia(self):
        """Si sensibilidad = 1 - especificidad, el modelo no aporta información."""
        prevalence = 0.02
        assert ppv_at_prevalence(0.5, 0.5, prevalence) == pytest.approx(prevalence)

    @pytest.mark.parametrize("prevalence", [-0.1, 1.5])
    def test_prevalencia_invalida(self, prevalence):
        with pytest.raises(ValueError, match="prevalencia"):
            ppv_at_prevalence(0.9, 0.9, prevalence)


class TestNPVAtPrevalence:
    def test_npv_alto_con_prevalencia_baja(self):
        """Con enfermedad rara, un negativo es casi con seguridad correcto."""
        assert npv_at_prevalence(0.90, 0.90, 0.02) > 0.99

    def test_clasificador_perfecto(self):
        assert npv_at_prevalence(1.0, 1.0, 0.05) == 1.0


class TestThresholdForSpecificity:
    def test_alcanza_la_especificidad_pedida(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(900), np.ones(100)])
        y_score = np.concatenate([rng.beta(2, 5, 900), rng.beta(5, 2, 100)])

        threshold = threshold_for_specificity(y_true, y_score, 0.95)
        op = operating_point_at_threshold(y_true, y_score, threshold, prevalence=0.02)

        assert op.specificity == pytest.approx(0.95, abs=0.02)

    def test_mayor_especificidad_implica_mayor_umbral(self):
        rng = np.random.default_rng(0)
        y_true = np.concatenate([np.zeros(500), np.ones(500)])
        y_score = np.concatenate([rng.beta(2, 5, 500), rng.beta(5, 2, 500)])

        assert threshold_for_specificity(y_true, y_score, 0.99) > threshold_for_specificity(
            y_true, y_score, 0.90
        )

    def test_sin_negativos(self):
        with pytest.raises(ValueError, match="negativas"):
            threshold_for_specificity(np.ones(10), np.random.rand(10), 0.95)


class TestClassificationReport:
    @pytest.fixture
    def datos_desbalanceados(self):
        """1.000 muestras con 2 % de prevalencia y un clasificador razonable."""
        rng = np.random.default_rng(123)
        n_pos, n_neg = 20, 980
        y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        y_score = np.concatenate([rng.beta(6, 2, n_pos), rng.beta(2, 6, n_neg)])
        return y_true, y_score

    def test_no_reporta_accuracy(self, datos_desbalanceados):
        """La accuracy está excluida a propósito: es engañosa con desbalanceo."""
        report = classification_report(*datos_desbalanceados)
        assert not any("acc" in key.lower() for key in report)

    def test_incluye_metricas_clave(self, datos_desbalanceados):
        report = classification_report(*datos_desbalanceados)
        assert "pr_auc" in report
        assert "operating_points" in report
        assert 0.0 <= report["pr_auc"] <= 1.0

    def test_un_punto_de_operacion_por_especificidad(self, datos_desbalanceados):
        report = classification_report(
            *datos_desbalanceados, target_specificities=(0.90, 0.95, 0.99)
        )
        assert len(report["operating_points"]) == 3

    def test_el_vpp_mejora_al_subir_la_especificidad(self, datos_desbalanceados):
        report = classification_report(*datos_desbalanceados, target_specificities=(0.90, 0.99))
        low, high = report["operating_points"]
        assert high.ppv > low.ppv

    def test_registra_ambas_prevalencias(self, datos_desbalanceados):
        """La del test y la de campo son distintas y ambas deben quedar registradas."""
        report = classification_report(*datos_desbalanceados, field_prevalence=0.03)
        assert report["test_prevalence"] == pytest.approx(0.02)
        assert report["field_prevalence"] == pytest.approx(0.03)

    def test_dimensiones_incompatibles(self):
        with pytest.raises(ValueError, match="Dimensiones"):
            classification_report(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))

    def test_etiquetas_no_binarias(self):
        with pytest.raises(ValueError, match="binario"):
            classification_report(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
