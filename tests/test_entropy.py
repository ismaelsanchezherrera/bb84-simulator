"""
Tests de EntropySource, centrados en el fix de v5.4 para choice() en modo
crypto: antes de esta versión, `size` como tupla (p.ej. (1, 1), o
cualquier forma multidimensional) fallaba con TypeError.
"""
import numpy as np
import pytest

from bb84_simulator import EntropySource


@pytest.fixture
def entropy_crypto():
    return EntropySource.crypto()


def test_choice_crypto_size_tupla_con_reemplazo(entropy_crypto):
    resultado = entropy_crypto.choice(np.arange(10), size=(2, 3), replace=True)
    assert resultado.shape == (2, 3)
    assert np.all((resultado >= 0) & (resultado < 10))


def test_choice_crypto_size_tupla_sin_reemplazo(entropy_crypto):
    resultado = entropy_crypto.choice(np.arange(10), size=(1, 1), replace=False)
    assert resultado.shape == (1, 1)


def test_choice_crypto_size_entero_sigue_funcionando(entropy_crypto):
    resultado = entropy_crypto.choice(np.arange(20), size=5, replace=False)
    assert resultado.shape == (5,)
    assert len(set(resultado.tolist())) == 5  # sin reemplazo => sin repetidos


def test_choice_crypto_a_entero_equivale_a_range(entropy_crypto):
    # Igual que np.random.Generator.choice: `a` entero = elegir de range(a).
    resultado = entropy_crypto.choice(10, size=(4,))
    assert np.all((resultado >= 0) & (resultado < 10))


def test_choice_crypto_sin_reemplazo_mas_elementos_que_poblacion_falla(entropy_crypto):
    with pytest.raises(ValueError):
        entropy_crypto.choice(np.arange(5), size=10, replace=False)


def test_choice_crypto_size_none_devuelve_escalar(entropy_crypto):
    resultado = entropy_crypto.choice(np.arange(10))
    assert resultado in range(10)


def test_permutation_crypto_es_permutacion_valida():
    entropy = EntropySource.crypto()
    perm = entropy.permutation(50)
    assert sorted(perm.tolist()) == list(range(50))


def test_integers_bits_camino_rapido_produce_solo_0_y_1():
    entropy = EntropySource.crypto()
    bits = entropy.integers(0, 2, size=10_000)
    assert set(np.unique(bits).tolist()) <= {0, 1}
