"""
Modelos de canal cuántico (Fibra óptica, Despolarizante y Espacio Libre).
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
    """Física de detección compartida por todos los ChannelModel."""
    eta_total = eta_fibra * eta_detector
    n = len(packet.bits)

    llega_foton = entropy.random(n) < eta_total
    hay_dark_count = entropy.random(n) < prob_dark_count
    hay_click = llega_foton | hay_dark_count
    doble_click = llega_foton & hay_dark_count

    bases_bob = bob.choose_bases(n)
    resultado_real = bob.measure(packet_canal, bases_bob)

    if qber_intrinseco > 0:
        flips = entropy.random(n) < qber_intrinseco
        resultado_real = np.where(flips, 1 - resultado_real, resultado_real)

    if prob_depolarizacion > 0:
        despolariza = entropy.random(n) < prob_depolarizacion
        aleatorio = entropy.integers(0, 2, size=n)
        resultado_real = np.where(despolariza, aleatorio, resultado_real)

    resultado_dark = entropy.integers(0, 2, size=n)
    bits_bob = np.where(
        doble_click,
        entropy.integers(0, 2, size=n),
        np.where(llega_foton, resultado_real, resultado_dark),
    )

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
        n_dark_counts=int(np.sum(hay_dark_count)),
        n_double_clicks=int(np.sum(doble_click)),
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