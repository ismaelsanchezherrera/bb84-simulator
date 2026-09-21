"""
Tests de SecurityLayer / SecurityReport y de las propiedades estadísticas
de la capa cuántica. Migrados de run_statistical_tests (v5.3) y ampliados
en v5.4 con uniformidad de bases e independencia bit-base.
"""
import numpy as np

from bb84_simulator import (
    UMBRAL_QBER_SEGURIDAD,
    Alice,
    BB84Simulator,
    BitErrorEstimate,
    DetectionResult,
    EntropySource,
    InterceptResendEve,
    PhaseErrorEstimate,
    SecurityLayer,
    SecurityParameters,
)


def _deteccion_minima(n=1000):
    return DetectionResult(
        bits_alice=np.zeros(n, dtype=int),
        bases_alice=np.zeros(n, dtype=int),
        bits_bob=np.zeros(n, dtype=int),
        bases_bob=np.zeros(n, dtype=int),
        eta_fibra=1.0,
        eta_detector=1.0,
        eta_total=1.0,
        n_enviados=n,
        n_clicks=n,
        n_fotones_detectados=n,
        n_dark_counts=0,
        n_double_clicks=0,
    )


# --------------------------------------------------------------------
# SecurityReport / SecurityLayer
# --------------------------------------------------------------------

def test_qber_medido_coincide_con_bit_error_value():
    sim = BB84Simulator(EntropySource.simulation(seed=1), SecurityParameters())
    resultado = sim.run(n_qubits=20_000)
    assert resultado.qber_medido == resultado.bit_error.value


def test_security_report_no_tiene_campo_expected_qber():
    sim = BB84Simulator(EntropySource.simulation(seed=1), SecurityParameters())
    resultado = sim.run(n_qubits=20_000)
    assert "expected_qber" not in type(resultado).__dataclass_fields__


def test_evaluate_and_build_aborta_si_lhl_no_deja_bits():
    # v5.3: antes de esta corrección, longitud_clave_final=0 se reportaba
    # como "Transmisión Segura exitosa"; v5.4 hereda el comportamiento
    # corregido y lo cubre con un test directo sobre SecurityLayer.
    layer = SecurityLayer(SecurityParameters())
    detection = _deteccion_minima(1000)
    pe_data = {
        "bit_error": BitErrorEstimate(0.05, n_muestra=100, n_poblacion=1000, epsilon=1e-10),
        "phase_error_bound": PhaseErrorEstimate(0.08),  # por debajo de UMBRAL_QBER_SEGURIDAD
        "n_verificacion": 100,
        "clave_alice_resto": np.zeros(900, dtype=int),
        "clave_bob_resto": np.zeros(900, dtype=int),
    }
    reporte = layer.evaluate_and_build(
        detection=detection,
        n_tamizada=1000,
        pe_data=pe_data,
        bits_revelados_ec=50,
        discrepancias=0,
        confirmacion_clave_ok=True,
        clave_alice_pa=np.array([], dtype=np.uint8),  # LHL dejó 0 bits
        clave_bob_pa=np.array([], dtype=np.uint8),
        distancia_km=10.0,
    )
    assert reporte.abortado is True
    assert "LHL no deja bits" in reporte.razon
    assert reporte.longitud_clave_final == 0


def test_calculate_lhl_length_es_cero_en_umbral_de_qber():
    layer = SecurityLayer(SecurityParameters())
    e_ph = PhaseErrorEstimate(UMBRAL_QBER_SEGURIDAD)
    assert layer.calculate_lhl_length(n_resto=10_000, e_ph=e_ph, leak_ec=100, tag_length=40) == 0


# --------------------------------------------------------------------
# Estadística de la capa cuántica (migrado de run_statistical_tests)
# --------------------------------------------------------------------

def test_sifting_ratio_ideal_es_aproximadamente_la_mitad():
    sim = BB84Simulator(EntropySource.simulation(seed=42), SecurityParameters())
    resultado = sim.run(n_qubits=100_000, qber_intrinseco=0.0, eta_detector=1.0)
    ratio = resultado.n_tamizada / 100_000
    assert 0.48 < ratio < 0.52


def test_intercept_resend_produce_qber_de_un_cuarto():
    sim = BB84Simulator(EntropySource.simulation(seed=42), SecurityParameters())
    resultado = sim.run(
        n_qubits=100_000,
        qber_intrinseco=0.0,
        eta_detector=1.0,
        eve=InterceptResendEve(),
        eve_fraction=1.0,
    )
    assert 0.23 < resultado.qber_medido < 0.27


def test_uniformidad_e_independencia_bit_base_de_alice():
    # v5.4: además de las medias/chi-cuadrado marginales de bits y bases
    # por separado, se comprueba independencia conjunta sobre la tabla
    # de contingencia 2x2 -- lo único que detectaría una fuente con bit
    # y base anti-correlacionados pero ambas marginales en ~0.5.
    alice = Alice(EntropySource.simulation(seed=123))
    n = 200_000
    paquete = alice.prepare(n)

    assert 0.49 < float(np.mean(paquete.bits)) < 0.51
    assert 0.49 < float(np.mean(paquete.bases)) < 0.51

    conteo_bits = np.bincount(paquete.bits, minlength=2)
    chi2_bits = float(np.sum((conteo_bits - n / 2) ** 2) / (n / 2))
    assert chi2_bits < 10.83  # 1 g.l., p<0.001

    conteo_bases = np.bincount(paquete.bases, minlength=2)
    chi2_bases = float(np.sum((conteo_bases - n / 2) ** 2) / (n / 2))
    assert chi2_bases < 10.83

    tabla = np.zeros((2, 2), dtype=int)
    np.add.at(tabla, (paquete.bits, paquete.bases), 1)
    esperado = tabla.sum(axis=1, keepdims=True) * tabla.sum(axis=0, keepdims=True) / tabla.sum()
    chi2_independencia = float(np.sum((tabla - esperado) ** 2 / esperado))
    assert chi2_independencia < 10.83
