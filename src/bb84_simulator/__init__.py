"""
Simulador BB84 - Distribución de Claves Cuánticas.
"""

__version__ = "5.4.1"

from bb84_simulator.attacks import (
    CollectiveAttack,
    EveStrategy,
    InterceptResendEve,
    PassiveEve,
    QuantumPacket,
    RandomEve,
)
from bb84_simulator.cascade import error_correction_cascade
from bb84_simulator.channels import (
    ChannelModel,
    DepolarizingChannel,
    FiberChannel,
    FreeSpaceChannel,
)
from bb84_simulator.classical import ClassicalLayer
from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import DetectionResult, SecurityReport
from bb84_simulator.quantum import Alice, Bob, QuantumLayer
from bb84_simulator.security import (
    UMBRAL_QBER_SEGURIDAD,
    BitErrorEstimate,
    PhaseErrorEstimate,
    SecurityLayer,
    SecurityParameters,
    cota_serfling_superior,
    entropia_binaria,
)
from bb84_simulator.simulator import BB84Simulator, run_statistical_tests

__all__ = [
    "BB84Simulator",
    "EntropySource",
    "SecurityParameters",
    "SecurityReport",
    "DetectionResult",
    "QuantumPacket",
    "ChannelModel",
    "FiberChannel",
    "DepolarizingChannel",
    "FreeSpaceChannel",
    "EveStrategy",
    "InterceptResendEve",
    "PassiveEve",
    "RandomEve",
    "CollectiveAttack",
    "Alice",
    "Bob",
    "QuantumLayer",
    "ClassicalLayer",
    "error_correction_cascade",
    "SecurityLayer",
    "BitErrorEstimate",
    "PhaseErrorEstimate",
    "UMBRAL_QBER_SEGURIDAD",
    "cota_serfling_superior",
    "entropia_binaria",
    "run_statistical_tests",
]
