"""
Estrategias de intromisión e intercepción (Eve) y paquetes cuánticos para el protocolo BB84.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from bb84_simulator.entropy import EntropySource


@dataclass(frozen=True)
class QuantumPacket:
    """
    Representa un conjunto de qubits en tránsito preparados en bits y bases.
    """

    bits: np.ndarray
    bases: np.ndarray


class EveStrategy(ABC):
    """Interfaz abstracta para estrategias de ataque de Eve."""

    @abstractmethod
    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float
    ) -> QuantumPacket:
        pass


class InterceptResendEve(EveStrategy):
    """Ataque activo Intercept-Resend."""

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction debe estar entre 0 y 1.")

        n = len(packet.bits)
        intercept = entropy.random(n) < fraction

        eve_bases = entropy.integers(0, 2, size=n)
        eve_random = entropy.integers(0, 2, size=n)

        eve_results = np.where(packet.bases == eve_bases, packet.bits, eve_random)
        bits_out = np.where(intercept, eve_results, packet.bits)
        bases_out = np.where(intercept, eve_bases, packet.bases)

        return QuantumPacket(bits=bits_out, bases=bases_out)


class PassiveEve(EveStrategy):
    """Eve no interactúa con los qubits en vuelo."""

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        return packet


class RandomEve(EveStrategy):
    """
    Decorador de activación probabilística sobre otra EveStrategy.
    """

    def __init__(
        self, estrategia_base: EveStrategy, probabilidad_activacion: float = 0.5
    ):
        if not 0.0 <= probabilidad_activacion <= 1.0:
            raise ValueError("probabilidad_activacion debe estar entre 0 y 1.")
        self.estrategia_base = estrategia_base
        self.probabilidad_activacion = probabilidad_activacion

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        activa = entropy.random(None) < self.probabilidad_activacion
        if activa:
            return self.estrategia_base.attack(packet, entropy, fraction)
        return packet


class CollectiveAttack(EveStrategy):
    """
    Placeholder de interfaz para ataques colectivos.
    """

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        raise NotImplementedError(
            "CollectiveAttack no es simulable a nivel de bits/bases. "
            "Ver el docstring de la clase para el enfoque correcto "
            "(endurecer la cota de seguridad, no simular la medida de Eve)."
        )