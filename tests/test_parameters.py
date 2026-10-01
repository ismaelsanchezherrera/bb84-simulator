"""
Tests de validación de SecurityParameters (__post_init__, v5.5).

Verifica que el objeto acote correctamente los parámetros de seguridad
composables (epsilon_pe, epsilon_pa, epsilon_cor), la eficiencia FEC
(fec_efficiency) y la longitud del tag de confirmación (tag_length).
"""
import math

import pytest

from bb84_simulator import SecurityParameters


def test_defaults_construyen_sin_error():
    params = SecurityParameters()
    assert params.epsilon_pe == pytest.approx(1e-10)
    assert params.epsilon_pa == pytest.approx(1e-10)
    assert params.epsilon_cor == pytest.approx(1e-10)
    assert params.tag_length is None


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_cor"])
@pytest.mark.parametrize("valor_invalido", [0.0, 1.0, -0.1, 1.1, 2.0])
def test_epsilon_fuera_de_rango_abierto_lanza_valueerror(nombre, valor_invalido):
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: valor_invalido})


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_cor"])
def test_epsilon_booleano_se_rechaza(nombre):
    # isinstance(True, numbers.Real) es True en Python -- hay que excluir
    # bool explícitamente o True/False colarían como epsilon "válidos".
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: True})


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_cor"])
@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf, -math.inf])
def test_epsilon_nan_o_inf_lanza_valueerror(nombre, valor_invalido):
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: valor_invalido})


def test_fec_efficiency_menor_que_uno_lanza_valueerror():
    with pytest.raises(ValueError):
        SecurityParameters(fec_efficiency=0.99)


def test_fec_efficiency_igual_a_uno_es_valido():
    assert SecurityParameters(fec_efficiency=1.0).fec_efficiency == 1.0


@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf])
def test_fec_efficiency_nan_o_inf_lanza_valueerror(valor_invalido):
    with pytest.raises(ValueError):
        SecurityParameters(fec_efficiency=valor_invalido)


@pytest.mark.parametrize("tag_length", [0, -1, -100])
def test_tag_length_no_positivo_lanza_valueerror(tag_length):
    with pytest.raises(ValueError):
        SecurityParameters(tag_length=tag_length)


def test_tag_length_no_entero_lanza_valueerror():
    with pytest.raises(ValueError):
        SecurityParameters(tag_length=64.5)


def test_tag_length_none_se_deriva_de_epsilon_cor():
    params = SecurityParameters(epsilon_cor=1e-12, tag_length=None)
    # tag_length_efectivo = ceil(-log2(1e-12)) = ceil(39.86...) = 40
    assert params.tag_length_efectivo == 40


def test_tag_length_explicito_valido():
    params = SecurityParameters(epsilon_cor=1e-12, tag_length=128)
    assert params.tag_length_efectivo == 128


def test_tag_length_insuficiente_contradiccion_epsilon_cor_lanza_valueerror():
    # tag_length=1 da una cota de colisión de 2^-1 = 0.5, lo cual contradice epsilon_cor=1e-12
    with pytest.raises(ValueError):
        SecurityParameters(epsilon_cor=1e-12, tag_length=1)