"""
Modelos de datos y estructuras de reporte para el simulador BB84.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    bits_alice: np.ndarray
    bases_alice: np.ndarray
    bits_bob: np.ndarray
    bases_bob: np.ndarray
    eta_fibra: float
    eta_detector: float
    eta_total: float
    n_enviados: int
    n_clicks: int
    n_fotones_detectados: int
    n_dark_counts: int
    n_double_clicks: int


@dataclass(frozen=True)
class SecurityReport:
    """Resultado inmutable una vez construido."""

    distancia_km: float | None
    n_qubits: int
    eta_fibra: float
    eta_detector: float
    eta_total: float
    n_clicks: int
    n_tamizada: int
    n_verificacion: int
    bit_error: Any
    phase_error_bound: Any
    leak_ec_real: int
    leak_ec_teorico: float
    discrepancias_tras_cascade: int
    autenticacion_ok: bool
    abortado: bool
    razon: str
    longitud_clave_final: int
    tasa_asintotica_bits_por_pulso: float
    tasa_empirica_bits_por_pulso: float
    detector_click_rate: float
    dark_click_rate: float
    double_click_rate: float
    clave_final_alice: np.ndarray
    clave_final_bob: np.ndarray

    @property
    def claves_coinciden(self) -> bool:
        return bool(np.array_equal(self.clave_final_alice, self.clave_final_bob))

    @property
    def qber_medido(self) -> float:
        return self.bit_error.value

    @property
    def f_ec_empirico(self) -> float:
        n_resto = self.n_tamizada - self.n_verificacion
        if n_resto <= 0 or self.bit_error.value <= 0:
            return float("nan")
        from bb84_simulator.simulator import entropia_binaria

        h = float(entropia_binaria(self.bit_error.value))
        if h <= 0:
            return float("nan")
        return float(self.leak_ec_real / (n_resto * h))