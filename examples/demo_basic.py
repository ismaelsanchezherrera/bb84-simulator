"""
Ejemplo de simulación simple del protocolo BB84.
"""

from bb84_simulator import BB84Simulator, EntropySource, SecurityParameters


def run_example():
    entropy = EntropySource.simulation(seed=42)
    params = SecurityParameters()
    sim = BB84Simulator(entropy, params)

    resultado = sim.run(n_qubits=100_000, distancia_km=10.0)
    print(f"Simulación completada con estado: {'ABORTADO' if resultado.abortado else 'EXITOSO'}")
    print(f"Clave final generada: {resultado.longitud_clave_final} bits")


if __name__ == "__main__":
    run_example()