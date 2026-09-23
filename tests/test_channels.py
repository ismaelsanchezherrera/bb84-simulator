"""
Tests de validación (v5.4) y de física, de FiberChannel, DepolarizingChannel y FreeSpaceChannel.
"""
import math

import pytest

from bb84_simulator import (
    BB84Simulator,
    DepolarizingChannel,
    EntropySource,
    FiberChannel,
    FreeSpaceChannel,
    SecurityParameters,
)


@pytest.fixture
def entropy():
    return EntropySource.simulation(seed=1)


# --------------------------------------------------------------------
# FiberChannel: validación
# --------------------------------------------------------------------

def test_fiber_channel_distancia_negativa_lanza_valueerror(entropy):
    # Antes de v5.4: una distancia negativa producía
    # eta_fibra = 10**(+algo) > 1, una "ganancia" físicamente absurda.
    with pytest.raises(ValueError):
        FiberChannel(entropy, distancia_km=-1.0)


@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf])
def test_fiber_channel_distancia_nan_o_inf_lanza_valueerror(entropy, valor_invalido):
    # v5.4.1: `valor < 0.0` no descarta NaN (nan < 0.0 es False, igual
    # que cualquier comparación ordinaria con NaN) ni +inf tiene
    # sentido físico como distancia.
    with pytest.raises(ValueError):
        FiberChannel(entropy, distancia_km=valor_invalido)


def test_fiber_channel_atenuacion_negativa_lanza_valueerror(entropy):
    with pytest.raises(ValueError):
        FiberChannel(entropy, atenuacion_db_km=-0.2)


@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf])
def test_fiber_channel_atenuacion_nan_o_inf_lanza_valueerror(entropy, valor_invalido):
    with pytest.raises(ValueError):
        FiberChannel(entropy, atenuacion_db_km=valor_invalido)


@pytest.mark.parametrize("nombre", [
    "eta_detector", "prob_dark_count", "qber_intrinseco", "eve_fraction",
])
@pytest.mark.parametrize("valor", [-0.1, 1.1])
def test_fiber_channel_probabilidades_fuera_de_rango(entropy, nombre, valor):
    with pytest.raises(ValueError):
        FiberChannel(entropy, **{nombre: valor})


def test_fiber_channel_valores_limite_son_validos(entropy):
    # 0.0 y 1.0 son válidos (intervalo cerrado): no deben lanzar.
    canal = FiberChannel(entropy, eta_detector=0.0, prob_dark_count=1.0)
    assert canal.eta_detector == 0.0


# --------------------------------------------------------------------
# DepolarizingChannel: validación
# --------------------------------------------------------------------

@pytest.mark.parametrize("nombre", [
    "prob_depolarizacion", "eta_detector", "prob_dark_count", "eve_fraction",
])
@pytest.mark.parametrize("valor", [-0.01, 1.5])
def test_depolarizing_channel_probabilidades_fuera_de_rango(entropy, nombre, valor):
    with pytest.raises(ValueError):
        DepolarizingChannel(entropy, **{nombre: valor})


def test_depolarizing_channel_eta_fibra_es_uno(entropy):
    # Sin modelo de pérdida por distancia: eta_fibra siempre 1.0.
    canal = DepolarizingChannel(entropy)
    assert canal.eta_fibra == 1.0


# --------------------------------------------------------------------
# FreeSpaceChannel: validación
# --------------------------------------------------------------------

@pytest.mark.parametrize("elevacion", [0.0, -10.0, 91.0])
def test_free_space_channel_elevacion_fuera_de_rango(entropy, elevacion):
    with pytest.raises(ValueError):
        FreeSpaceChannel(entropy, elevacion_grados=elevacion)


@pytest.mark.parametrize("nombre", ["atenuacion_cenital_db", "perdida_apuntamiento_db"])
def test_free_space_channel_atenuacion_negativa(entropy, nombre):
    with pytest.raises(ValueError):
        FreeSpaceChannel(entropy, **{nombre: -1.0})


@pytest.mark.parametrize("nombre", [
    "eta_detector", "prob_dark_count", "qber_intrinseco", "eve_fraction",
])
def test_free_space_channel_probabilidad_fuera_de_rango(entropy, nombre):
    with pytest.raises(ValueError):
        FreeSpaceChannel(entropy, **{nombre: 1.5})


def test_free_space_channel_elevacion_90_es_valida(entropy):
    canal = FreeSpaceChannel(entropy, elevacion_grados=90.0)
    assert canal.masa_de_aire == pytest.approx(1.0)


# --------------------------------------------------------------------
# Física del canal (migrado de run_statistical_tests v5.3, Test 3 y 5)
# --------------------------------------------------------------------

def test_atenuacion_de_fibra_coincide_con_formula_teorica():
    sim = BB84Simulator(EntropySource.simulation(seed=42), SecurityParameters())
    resultado = sim.run(
        n_qubits=100_000,
        distancia_km=10.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=0.0,
    )
    eta_teorica = 10 ** (-0.2 * 10 / 10)  # 0.6309...
    assert math.isclose(resultado.detector_click_rate, eta_teorica, rel_tol=0.05)


def test_dark_count_rate_domina_en_canal_saturado():
    # A 500 km / 0.2 dB/km, eta_fibra ~ 1e-10: casi ningún fotón real
    # sobrevive, así que casi todos los clicks vienen de dark counts.
    # Con 2 detectores independientes (D0 y D1), prob_click_total ~ 2 * prob_dark_count.
    sim = BB84Simulator(EntropySource.simulation(seed=42), SecurityParameters())
    resultado = sim.run(
        n_qubits=200_000,
        distancia_km=500.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=1e-3,
        qber_intrinseco=0.0,
    )
    # 2 * 1e-3 = 2e-3 debido a la presencia de dos detectores D0 y D1
    prob_esperada = 2e-3
    assert math.isclose(resultado.detector_click_rate, prob_esperada, rel_tol=0.25)