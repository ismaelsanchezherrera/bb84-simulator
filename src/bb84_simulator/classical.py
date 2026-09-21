"""
Capa de post-procesamiento clásico (Tamizado, Estimación de QBER y Amplificación de Privacidad).
"""

from __future__ import annotations

from typing import Any
import numpy as np

from bb84_simulator.cascade import error_correction_cascade
from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import DetectionResult
from bb84_simulator.security import (
    BitErrorEstimate,
    PhaseErrorEstimate,
    SecurityParameters,
    cota_serfling_superior,
)


def _convolucion_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_total = len(a) + len(b) - 1
    n_fft = 1 << (n_total - 1).bit_length()

    A = np.fft.rfft(a, n_fft)
    B = np.fft.rfft(b, n_fft)
    conv = np.fft.irfft(A * B, n_fft)[:n_total]

    redondeo = float(np.max(np.abs(conv - np.round(conv))))
    if redondeo > 1e-3:
        raise RuntimeError(f"Redondeo FFT sospechoso: {redondeo:.2e}")

    return np.round(conv).astype(np.int64)


class ClassicalLayer:
    """Capa 2: Responsable del post-procesamiento clásico de información."""

    def __init__(self, entropy: EntropySource, sec_params: SecurityParameters):
        self.entropy = entropy
        self.sec_params = sec_params

    def sifting(self, detection: DetectionResult) -> tuple[np.ndarray, np.ndarray]:
        mask = detection.bases_alice == detection.bases_bob
        return detection.bits_alice[mask].copy(), detection.bits_bob[mask].copy()

    def parameter_estimation(
        self,
        clave_alice: np.ndarray,
        clave_bob: np.ndarray,
        fraccion_verificacion: float = 0.15,
    ) -> dict[str, Any]:
        if len(clave_alice) != len(clave_bob):
            raise ValueError(
                f"Las claves de Alice y Bob deben tener la misma longitud "
                f"(obtenido: Alice={len(clave_alice)}, Bob={len(clave_bob)})"
            )

        if not 0.0 < fraccion_verificacion < 1.0:
            raise ValueError("fraccion_verificacion debe estar en el intervalo (0, 1).")

        n = len(clave_alice)
        if n == 0:
            raise ValueError("No hay bits tamizados para estimación.")

        n_verif = max(1, int(n * fraccion_verificacion))
        idx_verif = self.entropy.choice(np.arange(n), size=n_verif, replace=False)

        errores = clave_alice[idx_verif] != clave_bob[idx_verif]
        qber_puntual = float(np.mean(errores))

        qber_superior = cota_serfling_superior(
            qber_puntual, n_verif, n, self.sec_params.epsilon_pe
        )

        mascara_resto = np.ones(n, dtype=bool)
        mascara_resto[idx_verif] = False

        return {
            "bit_error": BitErrorEstimate(
                qber_puntual,
                n_muestra=n_verif,
                n_poblacion=n,
                epsilon=self.sec_params.epsilon_pe,
            ),
            "phase_error_bound": PhaseErrorEstimate(qber_superior),
            "n_verificacion": n_verif,
            "clave_alice_resto": clave_alice[mascara_resto],
            "clave_bob_resto": clave_bob[mascara_resto],
        }

    def error_correction_cascade(
        self,
        clave_alice: np.ndarray,
        clave_bob: np.ndarray,
        qber_estimado: float,
        n_pasadas: int = 4,
    ) -> tuple[np.ndarray, int, int]:
        return error_correction_cascade(
            self.entropy,
            clave_alice,
            clave_bob,
            qber_estimado=qber_estimado,
            n_pasadas=n_pasadas,
        )

    def privacy_amplification_toeplitz(
        self,
        clave: np.ndarray,
        longitud_salida: int,
        semilla_publica: int | None = None,
    ) -> np.ndarray:
        longitud_salida = max(0, int(longitud_salida))
        n = len(clave)
        if longitud_salida == 0 or n == 0:
            return np.array([], dtype=np.uint8)

        if semilla_publica is None:
            semilla_publica = self.entropy.random_seed_int()

        rng = np.random.default_rng(semilla_publica)
        semilla_toeplitz = rng.integers(0, 2, size=longitud_salida + n - 1)

        conv = _convolucion_fft(semilla_toeplitz.astype(np.int64), clave.astype(np.int64))
        y = conv[n - 1 : n - 1 + longitud_salida]
        return (y & 1).astype(np.uint8)

    def key_confirmation(self, clave_a: np.ndarray, clave_b: np.ndarray) -> bool:
        if len(clave_a) == 0 or len(clave_b) == 0:
            return False
        seed = self.entropy.random_seed_int()
        tag_length = self.sec_params.tag_length_efectivo
        tag_a = self.privacy_amplification_toeplitz(clave_a, tag_length, seed)
        tag_b = self.privacy_amplification_toeplitz(clave_b, tag_length, seed)
        return bool(np.array_equal(tag_a, tag_b))