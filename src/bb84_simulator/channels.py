"""
Modelos de canal cuántico (Fibra óptica, Despolarizante y Espacio Libre)
con física de detección de dos detectores independientes (D0 y D1).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

from bb84_simulator.attacks import EveStrategy, QuantumPacket
from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import DetectionResult
from bb84_simulator.validation import (
    validar_no_negativo as _validar_no_negativo,
    validar_probabilidad as _validar_prob01,
)

if TYPE_CHECKING:
    from bb84_simulator.quantum import Bob


class ChannelModel(ABC):
    """Interfaz abstracta para el canal físico cuántico."""

    @abstractmethod
    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        pass


def _detectar_en_bob(
    packet: QuantumPacket,
    packet_canal: QuantumPacket,
    bob: Bob,
    entropy: EntropySource,
    eta_fibra: float,
    eta_detector: float,
    prob_dark_count: float,
    qber_intrinseco: float = 0.0,
    prob_depolarizacion: float = 0.0,
) -> DetectionResult:
    """
    Física de detección realista compartida con dos detectores independientes (D0 y D1).

    Física del modelo:
    1. Atenuación del canal (eta_total = eta_fibra * eta_detector).
    2. Si bases coinciden: el fotón se dirige a D0 (si bit=0) o D1 (si bit=1).
    3. Si bases discrepan: el fotón se divide en el divisor de haz y elige D0 o D1 con prob. 0.5.
    4. Cuentas oscuras independientes en D0 y D1 (prob_dark_count por detector).
    5. Doble clic (ambos detectores activos): asignación aleatoria uniforme de bit.
    """
    eta_total = eta_fibra * eta_detector
    n = len(packet.bits)

    # 1. Transmisión/llegada del fotón al módulo de detección
    llega_foton = entropy.random(n) < eta_total

    # 2. Selección de bases en Bob
    bases_bob = bob.choose_bases(n)

    # 3. Ruido intrínseco / despolarización previa a la medición
    bits_transito = packet_canal.bits.copy()
    if qber_intrinseco > 0:
        flips = entropy.random(n) < qber_intrinseco
        bits_transito[flips] ^= 1

    if prob_depolarizacion > 0:
        despolariza = entropy.random(n) < prob_depolarizacion
        bits_transito[despolariza] = entropy.integers(0, 2, size=int(np.sum(despolariza)))

    # 4. Divisor de haz y polarizador: enrutamiento a D0 y D1
    coinciden = packet_canal.bases == bases_bob
    rand_bs = entropy.random(n) < 0.5

    signal_d0 = llega_foton & (
        (coinciden & (bits_transito == 0)) | (~coinciden & rand_bs)
    )
    signal_d1 = llega_foton & (
        (coinciden & (bits_transito == 1)) | (~coinciden & ~rand_bs)
    )

    # 5. Cuentas oscuras independientes en D0 y D1
    dark_d0 = entropy.random(n) < prob_dark_count
    dark_d1 = entropy.random(n) < prob_dark_count

    # 6. Disparo final de detectores
    click_d0 = signal_d0 | dark_d0
    click_d1 = signal_d1 | dark_d1

    # 7. Clasificación de eventos de detección
    hay_click = click_d0 | click_d1
    doble_click = click_d0 & click_d1

    bits_bob = np.zeros(n, dtype=np.uint8)
    bits_bob[click_d1 & ~click_d0] = 1

    # En caso de doble clic, el resultado es ambiguo (asignación aleatoria de bit)
    idx_dobles = np.flatnonzero(doble_click)
    if len(idx_dobles) > 0:
        bits_bob[idx_dobles] = entropy.integers(0, 2, size=len(idx_dobles))

    idx_click = np.flatnonzero(hay_click)
    return DetectionResult(
        bits_alice=packet.bits[idx_click],
        bases_alice=packet.bases[idx_click],
        bits_bob=bits_bob[idx_click],
        bases_bob=bases_bob[idx_click],
        eta_fibra=eta_fibra,
        eta_detector=eta_detector,
        eta_total=eta_total,
        n_enviados=n,
        n_clicks=len(idx_click),
        n_fotones_detectados=int(np.sum(llega_foton)),
        n_dark_counts=int(np.sum((dark_d0 | dark_d1) & ~llega_foton)),
        n_double_clicks=len(idx_dobles),
    )


class FiberChannel(ChannelModel):
    """Modelo de canal de fibra óptica con atenuación y ruido."""

    def __init__(
        self,
        entropy: EntropySource,
        distancia_km: float = 0.0,
        atenuacion_db_km: float = 0.2,
        eta_detector: float = 0.2,
        prob_dark_count: float = 1e-6,
        qber_intrinseco: float = 0.01,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        _validar_no_negativo("distancia_km", distancia_km)
        _validar_no_negativo("atenuacion_db_km", atenuacion_db_km)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("qber_intrinseco", qber_intrinseco)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.distancia_km = distancia_km
        self.atenuacion_db_km = atenuacion_db_km
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.qber_intrinseco = qber_intrinseco
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def eta_fibra(self) -> float:
        return 10 ** (-self.atenuacion_db_km * self.distancia_km / 10.0)

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet,
            packet_canal,
            bob,
            self.entropy,
            eta_fibra=self.eta_fibra,
            eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            qber_intrinseco=self.qber_intrinseco,
        )


class DepolarizingChannel(ChannelModel):
    """Canal despolarizante simétrico independiente de la distancia."""

    def __init__(
        self,
        entropy: EntropySource,
        prob_depolarizacion: float = 0.02,
        eta_detector: float = 1.0,
        prob_dark_count: float = 0.0,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        _validar_prob01("prob_depolarizacion", prob_depolarizacion)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.prob_depolarizacion = prob_depolarizacion
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def eta_fibra(self) -> float:
        return 1.0

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet,
            packet_canal,
            bob,
            self.entropy,
            eta_fibra=self.eta_fibra,
            eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            prob_depolarizacion=self.prob_depolarizacion,
        )


class FreeSpaceChannel(ChannelModel):
    """Canal simplificado para un enlace de espacio libre dependiente de la elevación."""

    def __init__(
        self,
        entropy: EntropySource,
        elevacion_grados: float = 45.0,
        atenuacion_cenital_db: float = 3.0,
        perdida_apuntamiento_db: float = 2.0,
        eta_detector: float = 0.5,
        prob_dark_count: float = 1e-5,
        qber_intrinseco: float = 0.01,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        if not 0.0 < elevacion_grados <= 90.0:
            raise ValueError("elevacion_grados debe estar en (0, 90].")
        _validar_no_negativo("atenuacion_cenital_db", atenuacion_cenital_db)
        _validar_no_negativo("perdida_apuntamiento_db", perdida_apuntamiento_db)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("qber_intrinseco", qber_intrinseco)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.elevacion_grados = elevacion_grados
        self.atenuacion_cenital_db = atenuacion_cenital_db
        self.perdida_apuntamiento_db = perdida_apuntamiento_db
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.qber_intrinseco = qber_intrinseco
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def masa_de_aire(self) -> float:
        return 1.0 / math.sin(math.radians(self.elevacion_grados))

    @property
    def eta_fibra(self) -> float:
        atenuacion_db = (
            self.atenuacion_cenital_db * self.masa_de_aire
            + self.perdida_apuntamiento_db
        )
        return 10 ** (-atenuacion_db / 10.0)

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet,
            packet_canal,
            bob,
            self.entropy,
            eta_fibra=self.eta_fibra,
            eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            qber_intrinseco=self.qber_intrinseco,
        )