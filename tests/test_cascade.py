"""
Tests de ClassicalLayer.error_correction_cascade: validación de entrada
(v5.4) y test de regresión de corrección de errores (heredado de
run_statistical_tests v5.3, Test 6).

Nota sobre n_pasadas=0 (ver también el docstring del método): NO produce,
en la práctica, un "éxito falso" silencioso en el pipeline completo de
BB84Simulator.run() -- key_confirmation() aborta después si las claves de
verdad difieren. Aun así se rechaza con ValueError porque desperdicia
cómputo, produce un resultado confuso, y no debería depender de que
key_confirmation() lo atrape para ser correcto.
"""
import numpy as np
import pytest

from bb84_simulator import ClassicalLayer, EntropySource, SecurityParameters


@pytest.fixture
def classical_layer():
    return ClassicalLayer(EntropySource.simulation(seed=7), SecurityParameters())


def test_n_pasadas_cero_lanza_valueerror(classical_layer):
    clave = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(clave, clave.copy(), qber_estimado=0.05, n_pasadas=0)


def test_n_pasadas_negativo_lanza_valueerror(classical_layer):
    clave = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(clave, clave.copy(), qber_estimado=0.05, n_pasadas=-1)


def test_n_pasadas_no_entero_lanza_valueerror_no_typeerror(classical_layer):
    # v5.4.1: antes solo se comprobaba `n_pasadas < 1`, así que
    # n_pasadas=1.5 pasaba esa cota (1.5 >= 1) y reventaba más abajo con
    # un TypeError interno de numpy/range al usarlo como tamaño de
    # bloque -- un fallo confuso en vez de un ValueError con mensaje
    # claro sobre qué parámetro está mal.
    clave = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(clave, clave.copy(), qber_estimado=0.05, n_pasadas=1.5)


def test_n_pasadas_booleano_se_rechaza(classical_layer):
    # bool es subclase de int en Python -- True/False no deben colarse
    # como n_pasadas "válidos".
    clave = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(clave, clave.copy(), qber_estimado=0.05, n_pasadas=True)


def test_longitudes_distintas_lanza_valueerror(classical_layer):
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(
            np.zeros(100, dtype=int), np.zeros(99, dtype=int), qber_estimado=0.05, n_pasadas=4
        )


@pytest.mark.parametrize("qber_invalido", [-0.01, 0.51, 1.0])
def test_qber_estimado_fuera_de_rango_lanza_valueerror(classical_layer, qber_invalido):
    clave = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        classical_layer.error_correction_cascade(
            clave, clave.copy(), qber_estimado=qber_invalido, n_pasadas=4
        )


def test_qber_estimado_cero_es_valido_no_divide_por_cero(classical_layer):
    # qber_estimado=0 NO provoca división por cero: ya existe la guarda
    # q = max(qber_estimado, 1/n) antes de calcular el tamaño de bloque.
    # Tampoco se "salta" Cascade: cero errores en la muestra de
    # verificación no implica cero errores en el resto de la clave.
    clave = np.zeros(100, dtype=int)
    _, bits_revelados, discrepancias = classical_layer.error_correction_cascade(
        clave, clave.copy(), qber_estimado=0.0, n_pasadas=4
    )
    assert discrepancias == 0
    assert bits_revelados > 0  # sigue anunciando paridades, aunque no haya errores


def test_claves_vacias_no_hace_nada(classical_layer):
    vacio = np.array([], dtype=int)
    bob_final, bits_revelados, discrepancias = classical_layer.error_correction_cascade(
        vacio, vacio.copy(), qber_estimado=0.05, n_pasadas=4
    )
    assert len(bob_final) == 0
    assert bits_revelados == 0
    assert discrepancias == 0


@pytest.mark.parametrize("qber_test", [0.01, 0.03, 0.05, 0.08])
def test_cascade_corrige_todos_los_errores_inyectados(qber_test):
    # Test de regresión (v5.3 Test 6): con muy pocos errores fijos y una
    # única semilla, un bug de Cascade con backtracking que no refresca
    # las paridades de pasadas futuras puede pasar desapercibido. Se
    # barre QBER realista para tener potencia estadística real.
    rng = np.random.default_rng(2026)
    n = 20_000
    alice = rng.integers(0, 2, size=n)
    bob = alice.copy()
    flips = rng.random(n) < qber_test
    bob[flips] ^= 1

    capa = ClassicalLayer(
        EntropySource.simulation(int(rng.integers(0, 2**31))), SecurityParameters()
    )
    _, _, discrepancias = capa.error_correction_cascade(
        alice, bob, qber_estimado=qber_test, n_pasadas=4
    )
    assert discrepancias == 0
