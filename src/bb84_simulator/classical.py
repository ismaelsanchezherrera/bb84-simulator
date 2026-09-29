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

    @staticmethod
    def privacy_amplification_toeplitz(
        key: np.ndarray, target_length: int, toeplitz_seed: np.ndarray
    ) -> np.ndarray:
        """
        Amplificación de privacidad mediante hash universal con matriz de Toeplitz binaria.
        Multiplicación exacta sobre GF(2) mediante stride_tricks sin depender de FFT ni SciPy.
        """
        n = len(key)
        if target_length <= 0 or n == 0:
            return np.array([], dtype=np.uint8)

        if target_length > n:
            raise ValueError(
                f"target_length ({target_length}) no puede ser mayor que la clave ({n})"
            )

        if len(toeplitz_seed) < n + target_length - 1:
            raise ValueError("Semilla Toeplitz insuficiente para las dimensiones dadas.")

        col = toeplitz_seed[:target_length]
        row = toeplitz_seed[target_length - 1 :]

        # Construcción nativa con NumPy en GF(2)
        vals = np.concatenate((row[-1:0:-1], col))
        stride = vals.strides[0]
        matrix = np.lib.stride_tricks.as_strided(
            vals[len(row) - 1 :],
            shape=(target_length, n),
            strides=(stride, -stride),
        )

        final_key = (matrix @ key) % 2
        return final_key.astype(np.uint8)

    @staticmethod
    def privacy_amplification(
        key: np.ndarray,
        target_length: int,
        entropy: EntropySource,
    ) -> np.ndarray:
        """Método estático de conveniencia que extrae entropía directa e invoca PA Toeplitz."""
        n = len(key)
        if target_length <= 0 or n == 0:
            return np.array([], dtype=np.uint8)

        num_bits = n + target_length - 1
        toeplitz_seed = entropy.raw_crypto_bits(num_bits)
        return ClassicalLayer.privacy_amplification_toeplitz(
            key, target_length, toeplitz_seed
        )

    def key_confirmation(
        self,
        clave_a: np.ndarray,
        clave_b: np.ndarray,
        semilla_publica: int | None = None,
    ) -> bool:
        """Verifica si las claves reconciliadas coinciden mediante confirmación por etiquetas Toeplitz."""
        if len(clave_a) == 0 or len(clave_b) == 0 or len(clave_a) != len(clave_b):
            return False

        tag_length = self.sec_params.tag_length_efectivo
        num_bits = len(clave_a) + tag_length - 1

        if semilla_publica is None:
            semilla_publica = self.entropy.random_seed_int()

        # Generamos la semilla determinista compartida como arreglo de bits
        rng_compartido = EntropySource.simulation(seed=int(semilla_publica))
        toeplitz_seed = rng_compartido.integers(0, 2, size=num_bits).astype(np.uint8)

        tag_a = self.privacy_amplification_toeplitz(clave_a, tag_length, toeplitz_seed)
        tag_b = self.privacy_amplification_toeplitz(clave_b, tag_length, toeplitz_seed)

        return bool(np.array_equal(tag_a, tag_b))