"""
Parámetros y evaluación de cotas de seguridad de la información (Serfling y Leftover Hash Lemma).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Optional
import numpy as np

from bb84_simulator.models import DetectionResult, SecurityReport
from bb84_simulator.validation import (
    validar_epsilon as _validar_epsilon,
)

UMBRAL_QBER_SEGURIDAD = 0.11


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
class PhaseErrorEstimate:
    """Cota superior del error de fase tras aplicar la desigualdad de Serfling."""

    value: float


@dataclass(frozen=True)
class SecurityParameters:
    """
    Configuración del marco de seguridad composable.
    
    Proporciona parámetros de cota de error para estimación de parámetros (epsilon_pe),
    amplificación de privacidad (epsilon_pa) y corrección/confirmación (epsilon_cor).
    Mantiene compatibilidad con alias históricos de la suite de pruebas.
    """

    epsilon_pe: float = 1e-10  # Cota de estimación de parámetros (Serfling)
    epsilon_pa: float = 1e-10  # Cota de secreto/seguridad (Toeplitz/LHL)
    epsilon_cor: float = 1e-10 # Cota de corrección/confirmación

    # Campos de compatibilidad con la suite existente para instanciación con kwargs
    epsilon_ec: float = 1e-10
    epsilon_auth: float = 1e-12
    epsilon_ec: float = 1e-10
    fec_efficiency: float = 1.10
    explicit_tag_length: Optional[int] = None
    tag_length: Optional[int] = None

    def __post_init__(self) -> None:
        # Sincronización de alias de compatibilidad con soporte retrocompatible
        if self.tag_length is not None and self.explicit_tag_length is None:
            object.__setattr__(self, "explicit_tag_length", self.tag_length)
        elif self.explicit_tag_length is not None and self.tag_length is None:
            object.__setattr__(self, "tag_length", self.explicit_tag_length)

        # Si el usuario configura epsilon_ec pero no epsilon_cor, los sincronizamos
        if self.epsilon_ec != 1e-10 and self.epsilon_cor == 1e-10:
            object.__setattr__(self, "epsilon_cor", self.epsilon_ec)
        elif self.epsilon_cor != 1e-10 and self.epsilon_ec == 1e-10:
            object.__setattr__(self, "epsilon_ec", self.epsilon_cor)
            
        # Validaciones de rigurosidad sobre las cotas epsilon exclusivas
        _validar_epsilon("epsilon_pe", self.epsilon_pe)
        _validar_epsilon("epsilon_pa", self.epsilon_pa)
        _validar_epsilon("epsilon_cor", self.epsilon_cor)
        _validar_epsilon("epsilon_ec", self.epsilon_ec)
        _validar_epsilon("epsilon_auth", self.epsilon_auth)

        if (
            isinstance(self.fec_efficiency, bool)
            or not isinstance(self.fec_efficiency, Real)
            or not math.isfinite(self.fec_efficiency)
            or self.fec_efficiency < 1.0
        ):
            raise ValueError("fec_efficiency debe ser un real finito >= 1.")

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
        """Tamaño en bits necesario para la etiqueta de confirmación."""
        if self.explicit_tag_length is not None:
            return self.explicit_tag_length
        return calcular_tag_length(self.epsilon_auth)

    @property
    def epsilon_total(self) -> float:
        """Cota de seguridad composable global (Cota Unión interna)."""
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

    @staticmethod
    def calculate_lhl_length(
        n_resto: int,
        e_ph: float | PhaseErrorEstimate,
        leak_ec: int,
        tag_length: int,
        epsilon_pa: float = 1e-10,
    ) -> int:
        """Calcula la longitud de clave segura mediante la Leftover Hash Lemma (LHL)."""
        if n_resto < 0:
            raise ValueError("n_resto no puede ser negativo")
        if not (0 <= leak_ec <= n_resto):
            raise ValueError(f"leak_ec ({leak_ec}) fuera del rango [0, {n_resto}]")
        if tag_length < 0:
            raise ValueError("tag_length no puede ser negativo")

        e_ph_val = float(e_ph.value) if hasattr(e_ph, "value") else float(e_ph)
        
        # Cierre de seguridad: Si la cota de error de fase alcanza o supera el umbral
        # tolerable (0.11), no se pueden extraer bits seguros mediante LHL.
        if e_ph_val >= UMBRAL_QBER_SEGURIDAD:
            return 0

        h2 = entropia_binaria(e_ph_val)
        delta_pa = 2.0 * np.log2(1.0 / epsilon_pa)
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
                leak_ec_real=bits_revelados_ec,  # <-- Corrección: Registrar la fuga real consumida
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