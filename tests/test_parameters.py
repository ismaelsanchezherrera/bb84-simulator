"""
Tests de validación de SecurityParameters (__post_init__, v5.4 / v5.4.1).

Antes de v5.4, SecurityParameters(epsilon_pa=2) se aceptaba sin más y
volvía negativo (indebidamente favorable) el término
2*log2(1/epsilon_pa) de calculate_lhl_length -- inflando la longitud de
clave declarada segura. Estos tests cubren esa familia de fallos de
validación ausente, uno por parámetro, más los casos de NaN/inf
encontrados en v4.4.1 (comparaciones de un solo lado que no descartan
NaN, ya que ninguna comparación ordinaria con NaN es verdadera).
"""
import math

import pytest

from bb84_simulator import SecurityParameters


def test_defaults_construyen_sin_error():
    params = SecurityParameters()
    assert params.epsilon_pe == pytest.approx(1e-10)
    assert params.tag_length is None


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_auth", "epsilon_ec"])
@pytest.mark.parametrize("valor_invalido", [0.0, 1.0, -0.1, 1.1, 2.0])
def test_epsilon_fuera_de_rango_abierto_lanza_valueerror(nombre, valor_invalido):
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: valor_invalido})


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_auth", "epsilon_ec"])
def test_epsilon_booleano_se_rechaza(nombre):
    # isinstance(True, numbers.Real) es True en Python -- hay que excluir
    # bool explícitamente o True/False colarían como epsilon "válidos".
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: True})


@pytest.mark.parametrize("nombre", ["epsilon_pe", "epsilon_pa", "epsilon_auth", "epsilon_ec"])
@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf, -math.inf])
def test_epsilon_nan_o_inf_lanza_valueerror(nombre, valor_invalido):
    # Aquí ya estaba protegido antes de v5.4.1 (comparación encadenada
    # acotada por ambos lados descarta NaN/inf por cortocircuito), pero
    # se deja como test explícito para que no se rompa sin darse cuenta
    # si alguien simplifica la validación en el futuro.
    with pytest.raises(ValueError):
        SecurityParameters(**{nombre: valor_invalido})


def test_fec_efficiency_menor_que_uno_lanza_valueerror():
    with pytest.raises(ValueError):
        SecurityParameters(fec_efficiency=0.99)


def test_fec_efficiency_igual_a_uno_es_valido():
    assert SecurityParameters(fec_efficiency=1.0).fec_efficiency == 1.0


@pytest.mark.parametrize("valor_invalido", [math.nan, math.inf])
def test_fec_efficiency_nan_o_inf_lanza_valueerror(valor_invalido):
    # v5.4.1: `fec_efficiency < 1.0` no descartaba NaN (nan < 1.0 es
    # False) ni +inf tenía sentido físico como factor de eficiencia.
    with pytest.raises(ValueError):
        SecurityParameters(fec_efficiency=valor_invalido)


@pytest.mark.parametrize("tag_length", [0, -1, -100])
def test_tag_length_no_positivo_lanza_valueerror(tag_length):
    with pytest.raises(ValueError):
        SecurityParameters(tag_length=tag_length)


def test_tag_length_no_entero_lanza_valueerror():
    with pytest.raises(ValueError):
        SecurityParameters(tag_length=64.5)


def test_tag_length_none_se_deriva_de_epsilon_auth():
    params = SecurityParameters(epsilon_auth=1e-12, tag_length=None)
    # tag_length_efectivo = ceil(log2(1/1e-12)) = ceil(39.86...) = 40
    assert params.tag_length_efectivo == 40


def test_tag_length_explicito_ignora_epsilon_auth():
    params = SecurityParameters(epsilon_auth=1e-12, tag_length=128)
    assert params.tag_length_efectivo == 128
