"""
Pruebas de propiedades criptográficas, multiplicación GF(2), Poisson y límites de seguridad.
"""

import numpy as np
import pytest

from bb84_simulator.classical import ClassicalLayer
from bb84_simulator.entropy import EntropySource
from bb84_simulator.security import SecurityParameters, entropia_binaria


def test_toeplitz_gf2_multiplication():
    """Verifica que la amplificación Toeplitz opere correctamente sobre GF(2)."""
    entropy = EntropySource.simulation(seed=42)
    key = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    target_len = 3

    compressed = ClassicalLayer.privacy_amplification(key, target_len, entropy)
    assert len(compressed) == target_len
    assert np.all(np.isin(compressed, [0, 1]))


def test_poisson_edge_cases():
    """Verifica el comportamiento de Poisson para valores 0, frontera y vectores."""
    entropy = EntropySource.simulation(seed=42)

    # Caso escalar lambda = 0
    assert entropy.poisson(0.0) == 0

    # Caso vectorizado con ceros y valores válidos
    res = entropy.poisson(np.array([0.0, 2.5, 0.0]))
    assert res[0] == 0 and res[2] == 0

    # Validación de errores
    with pytest.raises(ValueError):
        entropy.poisson(-1.0)

    with pytest.raises(ValueError):
        entropy.poisson(750.0)  # Supera límite de estabilidad 700


def test_entropia_binaria_domain():
    """Verifica la validación de rango para H2(p)."""
    assert entropia_binaria(0.0) == 0.0
    assert abs(entropia_binaria(0.5) - 1.0) < 1e-6

    with pytest.raises(ValueError):
        entropia_binaria(-0.1)

    with pytest.raises(ValueError):
        entropia_binaria(0.6)


def test_tag_length_validation():
    """Verifica que no se acepten tags explícitos incoherentes con epsilon_auth."""
    # Tag de 1 bit da cota 0.5, incompatible con epsilon_auth = 1e-12
    with pytest.raises(ValueError):
        SecurityParameters(epsilon_auth=1e-12, explicit_tag_length=1)

    # Tag de 40 bits da cota 2^-40 ~ 9e-13, compatible con epsilon_auth = 1e-12
    params = SecurityParameters(epsilon_auth=1e-12, explicit_tag_length=40)
    assert params.tag_length_efectivo == 40