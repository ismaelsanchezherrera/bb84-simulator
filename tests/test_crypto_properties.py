"""
Pruebas de propiedades criptográficas, multiplicación GF(2), Poisson y límites de seguridad.
"""

import numpy as np
import pytest
from scipy.linalg import toeplitz

from bb84_simulator.attacks import QuantumPacket
from bb84_simulator.channels import FiberChannel
from bb84_simulator.classical import ClassicalLayer
from bb84_simulator.entropy import EntropySource
from bb84_simulator.quantum import Bob
from bb84_simulator.security import SecurityLayer, SecurityParameters, entropia_binaria


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


# --- 1. TEST DE TOEPLITZ CONTRA MULTIPLICACIÓN MATRICIAL DIRECTA EN GF(2) ---
def test_toeplitz_gf2_exact_matrix_multiplication():
    """
    Comprueba que la implementación de amplificación de privacidad
    coincide bit a bit con la multiplicación directa M @ key (mod 2).
    Esto detecta cualquier offset o error de indexación sutil.
    """
    entropy_for_manual = EntropySource.simulation(seed=12345)
    key = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    n = len(key)
    target_len = 4

    num_bits = target_len + n - 1
    col_row_bits = entropy_for_manual.raw_crypto_bits(num_bits)

    # Construcción exacta de la matriz M_(target_len x n):
    # M[i, j] = col_row_bits[i - j + n - 1]
    col = col_row_bits[n - 1 : n - 1 + target_len]
    row = col_row_bits[n - 1 :: -1][:n]

    M_directa = toeplitz(col, row) % 2
    resultado_esperado = (M_directa @ key) % 2

    # Ejecutamos la función de la capa clásica con la misma semilla
    entropy_for_function = EntropySource.simulation(seed=12345)
    resultado_capa = ClassicalLayer.privacy_amplification(
        key, target_len, entropy_for_function
    )

    np.testing.assert_array_equal(resultado_capa, resultado_esperado)


# --- 2. TEST DE SERFLING CONTRA POBLACIONES FINITAS CONOCIDAS ---
def test_serfling_bound_finite_population_values():
    """
    Verifica que la cota de Serfling genera los desvíos matemáticos esperados.
    """
    qber_medido = 0.05
    n_sample = 1000
    n_key_resto = 9000
    n_poblacion = n_sample + n_key_resto  # N = 10000
    epsilon_pe = 1e-10

    bound = SecurityLayer.serfling_bound(qber_medido, n_sample, n_poblacion, epsilon_pe)

    assert bound > qber_medido
    assert 0.15 < bound < 0.18  # Cota matemática calculada ~0.1631

    # Prueba de límite 0 (sin muestra)
    bound_zero = SecurityLayer.serfling_bound(qber_medido, 0, n_poblacion, epsilon_pe)
    assert bound_zero == 1.0


# --- 3. TEST DE LA FÍSICA DE LOS DOS DETECTORES ---
def test_two_detector_physics_alignment():
    """
    Valida la física cuántica de la detección D0/D1:
    1. Si bases de Alice y Bob coinciden, el detector exacto hace clic (100% de match).
    2. Si bases discrepan, los fotones colapsan 50/50 independientemente del bit de Alice.
    """
    entropy = EntropySource.simulation(seed=42)
    canal = FiberChannel(
        entropy=entropy,
        distancia_km=0.0,
        atenuacion_db_km=0.0,
        eta_detector=1.0,
        prob_dark_count=0.0,
        qber_intrinseco=0.0,
    )

    n = 5000
    bits_alice = entropy.integers(0, 2, size=n).astype(np.uint8)
    bases_alice = entropy.integers(0, 2, size=n).astype(np.uint8)
    packet = QuantumPacket(bits=bits_alice, bases=bases_alice)

    # CASO 1: Bases coinciden al 100%
    bob_match = Bob(entropy)
    bob_match.choose_bases = lambda num: bases_alice.copy()

    det_match = canal.transmit(packet, bob_match)

    assert det_match.n_clicks == n  # Todos detectados (eta=1.0, sin dark counts)
    assert det_match.n_double_clicks == 0  # Sin ruido, sin dobles clics
    np.testing.assert_array_equal(bits_alice, det_match.bits_bob)  # QBER = 0.0

    # CASO 2: Bases discrepan al 100%
    bob_mismatch = Bob(entropy)
    bob_mismatch.choose_bases = lambda num: (1 - bases_alice).astype(np.uint8)

    det_mismatch = canal.transmit(packet, bob_mismatch)

    assert det_mismatch.n_clicks == n
    assert det_mismatch.n_double_clicks == 0

    # Al medir en la base ortogonal, el resultado debe ser completamente aleatorio (QBER ~ 50%)
    match_rate = float(np.mean(bits_alice == det_mismatch.bits_bob))
    assert 0.45 < match_rate < 0.55  # Matemáticamente converge a 0.5