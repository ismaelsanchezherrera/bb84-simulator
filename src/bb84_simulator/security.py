"""
Parámetros y evaluación de cotas de seguridad de la información (Serfling y Leftover Hash Lemma).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any
import numpy as np

from bb84_simulator.models import DetectionResult, SecurityReport
from bb84_simulator.validation import (
    validar_epsilon as _validar_epsilon,
)

UMBRAL_QBER_SEGURIDAD = 0.11


@dataclass(frozen=True)
class SecurityParameters:
    """Configuración del marco de seguridad composable."""

    epsilon_cor: float = 1e-10   # Cota de corrección (Cascade/EC)
    epsilon_sec: float = 1e-10   # Cota de secreto (LHL / Privacy Amplification)
    epsilon_auth: float = 1e-12  # Cota de fallo de autenticación/confirmación
    fec_efficiency: float = 1.10
    explicit_tag_length: int | None = None

    # Campos para compatibilidad con la suite existente
    epsilon_pe: float = 1e-10
    epsilon_pa: float = 1e-10
    epsilon_ec: float = 1e-10
    tag_length: int | None = None

    def __post_init__(self) -> None:
        # Sincronización de alias de compatibilidad
        if self.tag_length is not None and self.explicit_tag_length is None:
            object.__setattr__(self, "explicit_tag_length", self.tag_length)
        elif self.explicit_tag_length is not None and self.tag_length is None:
            object.__setattr__(self, "tag_length", self.explicit_tag_length)

        if self.epsilon_pe != 1e-10 and self.epsilon_sec == 1e-10:
            object.__setattr__(self, "epsilon_sec", self.epsilon_pe)
        elif self.epsilon_sec != 1e-10 and self.epsilon_pe == 1e-10:
            object.__setattr__(self, "epsilon_pe", self.epsilon_sec)

        if self.epsilon_pa != 1e-10 and self.epsilon_sec == 1e-10:
            object.__setattr__(self, "epsilon_sec", self.epsilon_pa)
        elif self.epsilon_sec != 1e-10 and self.epsilon_pa == 1e-10:
            object.__setattr__(self, "epsilon_pa", self.epsilon_sec)

        if self.epsilon_ec != 1e-10 and self.epsilon_cor == 1e-10:
            object.__setattr__(self, "epsilon_cor", self.epsilon_ec)
        elif self.epsilon_cor != 1e-10 and self.epsilon_ec == 1e-10:
            object.__setattr__(self, "epsilon_ec", self.epsilon_cor)

        # Validación de cotas epsilon
        _validar_epsilon("epsilon_cor", self.epsilon_cor)
        _validar_epsilon("epsilon_sec", self.epsilon_sec)
        _validar_epsilon("epsilon_auth", self.epsilon_auth)

        if (
            isinstance(self.fec_efficiency, bool)
            or not isinstance(self.fec_efficiency, Real)
            or not math.isfinite(self.fec_efficiency)
            or self.fec_efficiency < 1.0
        ):
            raise ValueError("fec_efficiency debe ser un real finito >= 1.")

        # Validación de coherencia de tag_length explícito frente a epsilon_auth
        tag = self.explicit_tag_length
        if tag is not None:
            if isinstance(tag, bool) or not isinstance(tag, int) or tag <= 0:
                raise ValueError("explicit_tag_length debe ser un entero positivo")
            cota_colision = 2.0 ** (-tag)
            if cota_colision > self.epsilon_auth:
                raise ValueError(
                    f"explicit_tag_length={tag} proporciona una cota de colisión "
                    f"de 2^-{tag} = {cota_colision:.2e}, "
                    f"lo cual contradice el epsilon_auth solicitado ({self.epsilon_auth:.2e})"
                )

    @property
    def tag_length_efectivo(self) -> int:
        """Calcula el tamaño necesario de la etiqueta basado en epsilon_auth."""
        if self.explicit_tag_length is not None:
            return self.explicit_tag_length
        return calcular_tag_length(self.epsilon_auth)

    @property
    def epsilon_total(self) -> float:
        """Cota de seguridad composable global (Cota Unión)."""
        return self.epsilon_cor + self.epsilon_sec + self.epsilon_auth

@dataclass(frozen=True)
class BitErrorEstimate:
    """Estimación del error de BIT (QBER), Serfling (1974)."""

    value: float
    n_muestra: int = 0
    n_poblacion: int = 0
    epsilon: float = 0.0


@dataclass(frozen=True)
class PhaseErrorEstimate:
    """Cota superior del error de FASE, e_ph."""

    value: float
    supuesto: str = (
        "e_ph := cota_serfling(qber_bit); canal simétrico, no derivado de forma independiente"
    )


def entropia_binaria(p: float | np.ndarray) -> float | np.ndarray:
    """Calcula la entropía binaria H2(p). Requiere p en el intervalo [0.0, 0.5]."""
    p_arr = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p_arr)) or np.any((p_arr < 0.0) | (p_arr > 0.5)):
        raise ValueError(
            f"El parámetro de error p debe estar en el intervalo [0.0, 0.5] (obtenido: {p})"
        )

    out = np.zeros_like(p_arr)
    mask = (p_arr > 0.0) & (p_arr <= 0.5)
    out[mask] = (
        -p_arr[mask] * np.log2(p_arr[mask])
        - (1.0 - p_arr[mask]) * np.log2(1.0 - p_arr[mask])
    )
    if np.ndim(p) == 0:
        return float(out)
    return out

def calcular_tag_length(epsilon_auth: float) -> int:
    """Calcula el tamaño necesario de tag evitando OverflowError en 1/epsilon_auth."""
    if not (0.0 < epsilon_auth < 1.0) or not np.isfinite(epsilon_auth):
        raise ValueError(
            f"epsilon_auth debe estar en el intervalo (0.0, 1.0) (obtenido: {epsilon_auth})"
        )
    return int(np.ceil(-np.log2(epsilon_auth)))


def cota_serfling_superior(
    q_estimado: float,
    n_muestra: int,
    n_poblacion: int,
    epsilon: float = 1e-10,
) -> float:
    """Cota superior de Serfling (1974) sobre el resto no muestreado."""
    if n_muestra <= 0 or n_poblacion <= 0:
        return 1.0
    if n_muestra >= n_poblacion:
        return float(np.clip(q_estimado, 0.0, 1.0))
    if not 0 < epsilon < 1:
        raise ValueError("epsilon debe estar entre 0 y 1.")

    factor = (n_poblacion - n_muestra + 1) / n_poblacion
    margen_poblacion = np.sqrt(factor * np.log(1.0 / epsilon) / (2.0 * n_muestra))
    margen_resto = margen_poblacion * n_poblacion / (n_poblacion - n_muestra)
    return float(min(1.0, q_estimado + margen_resto))


class SecurityLayer:
    """Capa 3: Evalúa la cota de información y genera el reporte final de seguridad."""

    def __init__(self, sec_params: SecurityParameters):
        self.sec_params = sec_params

    @staticmethod
    def serfling_bound(
        q_estimado: float,
        n_muestra: int,
        n_poblacion: int,
        epsilon: float = 1e-10,
    ) -> float:
        """Cota superior de Serfling (1974) sobre el resto no muestreado."""
        return cota_serfling_superior(q_estimado, n_muestra, n_poblacion, epsilon)

    def calculate_lhl_length(
        self,
        n_resto: int,
        e_ph: PhaseErrorEstimate,
        leak_ec: int,
        tag_length: int = 0,
    ) -> int:
        """Leftover Hash Lemma (LHL) para Finite-Key."""
        if n_resto <= 0 or e_ph.value >= UMBRAL_QBER_SEGURIDAD:
            return 0

        h2_eph = float(entropia_binaria(e_ph.value))
        term_entropia = n_resto * (1.0 - h2_eph)
        term_eps = 2.0 * math.log2(1.0 / self.sec_params.epsilon_pa)

        l_val = math.floor(term_entropia - leak_ec - tag_length - term_eps)
        return max(0, l_val)

    def evaluate_and_build(
        self,
        detection: DetectionResult,
        n_tamizada: int,
        pe_data: dict[str, Any] | None,
        bits_revelados_ec: int,
        discrepancias: int,
        confirmacion_clave_ok: bool,
        clave_alice_pa: np.ndarray,
        clave_bob_pa: np.ndarray,
        distancia_km: float | None,
        abort_reason: str | None = None,
    ) -> SecurityReport:
        n_enviados = detection.n_enviados
        click_rate = detection.n_clicks / n_enviados if n_enviados > 0 else 0.0
        dark_rate = detection.n_dark_counts / n_enviados if n_enviados > 0 else 0.0
        double_rate = detection.n_double_clicks / n_enviados if n_enviados > 0 else 0.0

        if abort_reason or pe_data is None:
            if pe_data is not None:
                bit_err_abort = pe_data["bit_error"]
                phase_bound_abort = pe_data["phase_error_bound"]
                n_verif_abort = pe_data["n_verificacion"]
            else:
                bit_err_abort = BitErrorEstimate(0.0)
                phase_bound_abort = PhaseErrorEstimate(1.0)
                n_verif_abort = 0

            return SecurityReport(
                distancia_km=distancia_km,
                n_qubits=n_enviados,
                eta_fibra=detection.eta_fibra,
                eta_detector=detection.eta_detector,
                eta_total=detection.eta_total,
                n_clicks=detection.n_clicks,
                n_tamizada=n_tamizada,
                n_verificacion=n_verif_abort,
                bit_error=bit_err_abort,
                phase_error_bound=phase_bound_abort,
                leak_ec_real=0,
                leak_ec_teorico=0.0,
                discrepancias_tras_cascade=discrepancias,
                confirmacion_clave_ok=False,
                abortado=True,
                razon=abort_reason or "Abortado.",
                longitud_clave_final=0,
                tasa_asintotica_bits_por_pulso=0.0,
                tasa_empirica_bits_por_pulso=0.0,
                detector_click_rate=click_rate,
                dark_click_rate=dark_rate,
                double_click_rate=double_rate,
                clave_final_alice=np.array([], dtype=np.uint8),
                clave_final_bob=np.array([], dtype=np.uint8),
            )

        bit_err = pe_data["bit_error"]
        phase_bound = pe_data["phase_error_bound"]
        n_resto = len(pe_data["clave_alice_resto"])

        leak_ec_real = bits_revelados_ec
        leak_ec_teorico = (
            self.sec_params.fec_efficiency
            * n_resto
            * float(entropia_binaria(bit_err.value))
        )

        abort = False
        razon = "Transmisión Segura exitosa."

        if phase_bound.value >= UMBRAL_QBER_SEGURIDAD:
            abort = True
            razon = f"QBER Cota ({phase_bound.value:.4f}) supera umbral ({UMBRAL_QBER_SEGURIDAD})."
        elif not confirmacion_clave_ok:
            abort = True
            razon = "Fallo de confirmación de clave."
        elif len(clave_alice_pa) == 0:
            abort = True
            razon = (
                "LHL no deja bits seguros (longitud final = 0): "
                "n_resto insuficiente para el leak_ec + tag + término de "
                "suavizado con este QBER."
            )

        final_len = len(clave_alice_pa) if not abort else 0
        tasa_empirica = final_len / n_enviados if n_enviados > 0 else 0.0

        h_q = float(entropia_binaria(bit_err.value))
        tasa_asintotica = (n_tamizada / n_enviados) * max(
            0.0, 1.0 - (1.0 + self.sec_params.fec_efficiency) * h_q
        )

        return SecurityReport(
            distancia_km=distancia_km,
            n_qubits=n_enviados,
            eta_fibra=detection.eta_fibra,
            eta_detector=detection.eta_detector,
            eta_total=detection.eta_total,
            n_clicks=detection.n_clicks,
            n_tamizada=n_tamizada,
            n_verificacion=pe_data["n_verificacion"],
            bit_error=bit_err,
            phase_error_bound=phase_bound,
            leak_ec_real=leak_ec_real,
            leak_ec_teorico=leak_ec_teorico,
            discrepancias_tras_cascade=discrepancias,
            confirmacion_clave_ok=confirmacion_clave_ok,
            abortado=abort,
            razon=razon,
            longitud_clave_final=final_len,
            tasa_asintotica_bits_por_pulso=tasa_asintotica,
            tasa_empirica_bits_por_pulso=tasa_empirica,
            detector_click_rate=click_rate,
            dark_click_rate=dark_rate,
            double_click_rate=double_rate,
            clave_final_alice=clave_alice_pa if not abort else np.array([], dtype=np.uint8),
            clave_final_bob=clave_bob_pa if not abort else np.array([], dtype=np.uint8),
        )

    