"""
Actores cuánticos (Alice, Bob) y capa de ejecución de la transmisión (QuantumLayer).
"""

from __future__ import annotations

import numpy as np

from bb84_simulator.attacks import QuantumPacket
from bb84_simulator.channels import ChannelModel
from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import DetectionResult


class Alice:
    """Emisor cuántico BB84."""

    def __init__(self, entropy: EntropySource):
        self.entropy = entropy

    def prepare(self, n_qubits: int) -> QuantumPacket:
        bits = self.entropy.integers(0, 2, size=n_qubits)
        bases = self.entropy.integers(0, 2, size=n_qubits)
        return QuantumPacket(bits=bits, bases=bases)


class Bob:
    """Receptor cuántico BB84."""

    def __init__(self, entropy: EntropySource):
        self.entropy = entropy

    def choose_bases(self, n_qubits: int) -> np.ndarray:
        return self.entropy.integers(0, 2, size=n_qubits)

    def measure(self, packet: QuantumPacket, bases_bob: np.ndarray) -> np.ndarray:
        coinciden = packet.bases == bases_bob
        aleatorios = self.entropy.integers(0, 2, size=len(packet.bits))
        return np.where(coinciden, packet.bits, aleatorios)


class QuantumLayer:
    """Capa 1: Responsable de la ejecución del hardware físico cuántico."""

    def __init__(self, alice: Alice, bob: Bob, channel: ChannelModel):
        self.alice = alice
        self.bob = bob
        self.channel = channel

    def execute(self, n_qubits: int) -> DetectionResult:
        packet = self.alice.prepare(n_qubits)
        return self.channel.transmit(packet, self.bob)