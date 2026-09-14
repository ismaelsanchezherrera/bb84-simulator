"""
Tests de integración de BB84Simulator.run(), con foco en la inyección de
`channel` añadida en v5.4 (antes exigía orquestar QuantumLayer +
ClassicalLayer + SecurityLayer a mano para usar un canal que no fuera de
fibra).
"""
import pytest

from bb84_simulator import (
    BB84Simulator,
    DepolarizingChannel,
    EntropySource,
    FreeSpaceChannel,
    InterceptResendEve,
    SecurityParameters,
)


def test_n_qubits_no_positivo_lanza_valueerror():
    sim = BB84Simulator(EntropySource.simulation(seed=1))
    with pytest.raises(ValueError):
        sim.run(n_qubits=0)


def test_run_por_defecto_construye_fiber_channel_y_reporta_distancia():
    sim = BB84Simulator(EntropySource.simulation(seed=1), SecurityParameters())
    resultado = sim.run(n_qubits=300_000, distancia_km=15.0, eta_detector=0.25)
    assert resultado.distancia_km == 15.0
    assert not resultado.abortado
    assert resultado.claves_coinciden


def test_run_con_espia_intercept_resend_completo_aborta():
    sim = BB84Simulator(EntropySource.simulation(seed=1), SecurityParameters())
    resultado = sim.run(
        n_qubits=300_000,
        distancia_km=15.0,
        eta_detector=0.25,
        eve=InterceptResendEve(),
    )
    assert resultado.abortado
    assert "QBER" in resultado.razon


def test_run_con_channel_y_parametro_de_fibra_no_default_lanza_valueerror():
    # v5.4: pasar `channel` junto con un parámetro propio de FiberChannel
    # con un valor distinto de su default se rechaza en vez de
    # ignorarse en silencio.
    sim = BB84Simulator(EntropySource.simulation(seed=1), SecurityParameters())
    canal = DepolarizingChannel(EntropySource.simulation(seed=2), prob_depolarizacion=0.02)
    with pytest.raises(ValueError):
        sim.run(n_qubits=5_000, channel=canal, distancia_km=10.0)
    with pytest.raises(ValueError):
        sim.run(n_qubits=5_000, channel=canal, eve=InterceptResendEve())


def test_run_con_channel_depolarizing_no_reporta_distancia():
    entropy = EntropySource.simulation(seed=3)
    sim = BB84Simulator(entropy, SecurityParameters())
    canal = DepolarizingChannel(EntropySource.simulation(seed=4), prob_depolarizacion=0.02, eta_detector=1.0)
    resultado = sim.run(n_qubits=150_000, channel=canal)
    assert resultado.distancia_km is None
    assert not resultado.abortado


def test_run_con_channel_freespace_no_reporta_distancia():
    entropy = EntropySource.simulation(seed=5)
    sim = BB84Simulator(entropy, SecurityParameters())
    canal = FreeSpaceChannel(EntropySource.simulation(seed=6), elevacion_grados=60.0)
    resultado = sim.run(n_qubits=150_000, channel=canal)
    assert resultado.distancia_km is None


def test_run_insuficientes_bits_tamizados_aborta_sin_reventar():
    sim = BB84Simulator(EntropySource.simulation(seed=7), SecurityParameters())
    resultado = sim.run(n_qubits=10, n_min_tamizados=100)
    assert resultado.abortado
    assert resultado.razon == "Insuficientes bits tamizados."
    assert resultado.longitud_clave_final == 0
