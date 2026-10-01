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


def entropia_binaria(p: float) -> float:
    """Calcula la entropía binaria de Shannon H2(p) = -p log2(p) - (1-p) log2(1-p)."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
        raise ValueError("p debe ser un número real finito.")
    if not (0.0 <= p <= 0.5):
        raise ValueError(f"p debe estar en el rango [0.0, 0.5]. Obtenido: {p}")
    if p == 0.0:
        return 0.0
    if p == 0.5:
        return 1.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def calcular_tag_length(epsilon_auth: float) -> int:
    """Calcula la longitud en bits de la etiqueta de confirmación para garantizar epsilon_auth."""
    _validar_epsilon("epsilon_auth", epsilon_auth)
    return int(np.ceil(-np.log2(epsilon_auth)))


@dataclass(frozen=True)
class BitErrorEstimate:
    """Estimación del error de bit obtenida en la etapa de verificación."""

    value: float
    n_muestra: int = 0
    n_poblacion: int = 0
    epsilon: float = 1e-10


@dataclass(frozen=True)
class SymmetricChannelPhaseErrorBound:
    """Cota superior del error de fase tras aplicar Serfling (asumiendo e_x ≈ e_z)."""

    value: float


@dataclass(frozen=True)
class SecurityParameters:
    """Configuración del marco de seguridad composable."""

    epsilon_pe: float = 1e-10  # Para Parameter Estimation
    epsilon_pa: float = 1e-10  # Para Privacy Amplification (LHL)
    epsilon_cor: float = 1e-10 # Para Correctness (Key Confirmation)
    fec_efficiency: float = 1.10
    tag_length: int | None = None

    def __post_init__(self) -> None:
        _validar_epsilon("epsilon_pe", self.epsilon_pe)
        _validar_epsilon("epsilon_pa", self.epsilon_pa)
        _validar_epsilon("epsilon_cor", self.epsilon_cor)

        if (
            isinstance(self.fec_efficiency, bool)
            or not isinstance(self.fec_efficiency, Real)
            or not math.isfinite(self.fec_efficiency)
            or self.fec_efficiency < 1.0
        ):
            raise ValueError("fec_efficiency debe ser un real finito >= 1.")

        if self.tag_length is not None:
            if isinstance(self.tag_length, bool) or not isinstance(self.tag_length, int) or self.tag_length <= 0:
                raise ValueError("tag_length debe ser un entero positivo")
            cota_colision = 2.0 ** (-self.tag_length)
            if cota_colision > self.epsilon_cor:
                raise ValueError(
                    f"tag_length={self.tag_length} da una cota de colisión de {cota_colision:.2e}, "
                    f"contradiciendo epsilon_cor ({self.epsilon_cor:.2e})"
                )

    @property
    def tag_length_efectivo(self) -> int:
        """Calcula el tamaño del tag de confirmación t = ceil(-log2(epsilon_cor))."""
        if self.tag_length is not None:
            return self.tag_length
        return int(np.ceil(-np.log2(self.epsilon_cor)))

    @property
    def epsilon_total(self) -> float:
        """Cota de seguridad composable global interna (epsilon_pe + epsilon_pa + epsilon_cor)."""
        return self.epsilon_pe + self.epsilon_pa + self.epsilon_cor


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
        e_ph: float | SymmetricChannelPhaseErrorBound,
        leak_ec: int,
        tag_length: int,
    ) -> int:
        """Calcula la longitud de clave segura (LHL) garantizando fronteras estrictas."""
        if n_resto < 0:
            raise ValueError("n_resto no puede ser negativo")
        if leak_ec < 0:
            raise ValueError("leak_ec no puede ser negativo")
        if not (0 <= tag_length <= n_resto):
            raise ValueError(f"tag_length ({tag_length}) fuera del rango [0, {n_resto}]")
        
        e_ph_val = float(e_ph.value) if hasattr(e_ph, "value") else float(e_ph)

        # Si el error de fase es >= 0.5 o la fuga de Cascade supera el bloque disponible,
        # es físicamente imposible extraer bits seguros.
        if e_ph_val >= 0.5 or leak_ec > n_resto:
            return 0

        eps_pa = self.sec_params.epsilon_pa
        if not (0.0 < eps_pa < 1.0):
            raise ValueError(f"epsilon_pa debe estar en (0, 1). Obtenido: {eps_pa}")

        h2 = entropia_binaria(e_ph_val)
        delta_pa = 2.0 * np.log2(1.0 / eps_pa)
        longitud_raw = (n_resto * (1.0 - h2)) - leak_ec - delta_pa - tag_length
        return max(0, int(np.floor(longitud_raw)))


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
                phase_bound_abort = SymmetricChannelPhaseErrorBound(1.0)
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
                leak_ec_real=bits_revelados_ec,
                leak_ec_teorico=float(bits_revelados_ec),
                discrepancias_tras_cascade=discrepancias,
                confirmacion_clave_ok=confirmacion_clave_ok,
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

        if not confirmacion_clave_ok:
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