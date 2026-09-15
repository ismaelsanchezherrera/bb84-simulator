"""
Punto de entrada CLI para el Simulador BB84.
"""

import sys
from bb84_simulator.simulator import (
    BB84Simulator,
    DepolarizingChannel,
    EntropySource,
    FreeSpaceChannel,
    InterceptResendEve,
    SecurityParameters,
    run_statistical_tests,
)


def main():
    # 1. Validar la integridad estadística del código
    run_statistical_tests()

    # 2. Configurar entorno de prueba
    entropy = EntropySource.simulation(seed=2026)
    security_params = SecurityParameters(
        epsilon_pe=1e-10,
        epsilon_pa=1e-10,
        epsilon_auth=1e-12,
        fec_efficiency=1.12,
    )

    simulador = BB84Simulator(entropy, security_params)

    print("--- Ejecutando Simulación BB84 V5.4 (canal sano, sin espía) ---")
    resultado = simulador.run(
        n_qubits=300_000,
        distancia_km=15.0,
        atenuacion_db_km=0.2,
        eta_detector=0.25,
        prob_dark_count=1e-6,
        qber_intrinseco=0.015,
        eve=None,
    )

    print(f"Estado de Ejecución:           {'ABORTADO' if resultado.abortado else 'EXITOSO'}")
    print(f"Razón:                         {resultado.razon}")
    print(f"QBER Estimado (Bit Error):     {resultado.bit_error.value * 100:.2f}%")
    print(f"Cota Superior Error Fase:      {resultado.phase_error_bound.value * 100:.2f}%")
    print(f"Fuga EC Real (Cascade):        {resultado.leak_ec_real} bits")
    print(f"Fuga EC Teórica (f_EC*H):      {resultado.leak_ec_teorico:.2f} bits")
    print(f"f_EC empírico vs. referencia:  {resultado.f_ec_empirico:.3f} "
          f"(referencia={security_params.fec_efficiency})")
    print(f"Tag de confirmación:           {security_params.tag_length_efectivo} bits "
          f"(derivado de epsilon_auth={security_params.epsilon_auth:.0e})")
    print(f"Longitud Clave Final (LHL):    {resultado.longitud_clave_final} bits")
    print(f"Claves Coinciden (Alice-Bob):  {resultado.claves_coinciden}")
    print(f"Click Rate Detectores:         {resultado.detector_click_rate * 100:.2f}%")
    print(f"Dark Count Rate:               {resultado.dark_click_rate * 100:.4f}%")

    print("\n--- Mismo canal, CON espía Intercept-Resend (debe abortar) ---")
    resultado_eve = simulador.run(
        n_qubits=300_000,
        distancia_km=15.0,
        atenuacion_db_km=0.2,
        eta_detector=0.25,
        prob_dark_count=1e-6,
        qber_intrinseco=0.015,
        eve=InterceptResendEve(),
    )
    print(f"Estado de Ejecución:           {'ABORTADO' if resultado_eve.abortado else 'EXITOSO'}")
    print(f"Razón:                         {resultado_eve.razon}")
    print(f"QBER Estimado (Bit Error):     {resultado_eve.bit_error.value * 100:.2f}%")

    print("\n--- DepolarizingChannel (sin modelo de pérdida por distancia) ---")
    entropy_depol = EntropySource.simulation(seed=7)
    canal_depol = DepolarizingChannel(entropy_depol, prob_depolarizacion=0.03, eta_detector=1.0)
    sim_depol = BB84Simulator(entropy_depol, security_params)
    resultado_depol = sim_depol.run(n_qubits=150_000, channel=canal_depol)
    print(f"n_tamizada: {resultado_depol.n_tamizada} | "
          f"QBER medido: {resultado_depol.qber_medido:.4f} "
          f"(prob_depolarizacion/2 = {0.03/2:.4f} esperado) | "
          f"distancia_km en el reporte: {resultado_depol.distancia_km}")

    print("\n--- FreeSpaceChannel (enlace de espacio libre, elevación 60°) ---")
    entropy_fs = EntropySource.simulation(seed=11)
    canal_fs = FreeSpaceChannel(entropy_fs, elevacion_grados=60.0)
    sim_fs = BB84Simulator(entropy_fs, security_params)
    resultado_fs = sim_fs.run(n_qubits=150_000, channel=canal_fs)
    print(f"Transmitancia atmosférica (eta_fibra): {canal_fs.eta_fibra:.4f} | "
          f"n_clicks: {resultado_fs.n_clicks} | "
          f"distancia_km en el reporte: {resultado_fs.distancia_km}")


if __name__ == "__main__":
    sys.exit(main())