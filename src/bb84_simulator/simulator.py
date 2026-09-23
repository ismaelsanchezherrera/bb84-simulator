"""
Simulador BB84 (Distribución de Claves Cuánticas).

Módulo principal para la simulación del protocolo QKD BB84, incluyendo
modelado de canales cuánticos con ruido, detección de intromisión (Eve),
reconciliación de información por Cascade y amplificación de privacidad.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from bb84_simulator.validation import (
    validar_entero_positivo,
    validar_no_negativo,
    validar_probabilidad,
)
from bb84_simulator.attacks import EveStrategy, InterceptResendEve
from bb84_simulator.channels import ChannelModel, FiberChannel
from bb84_simulator.classical import ClassicalLayer
from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import SecurityReport
from bb84_simulator.quantum import Alice, Bob, QuantumLayer
from bb84_simulator.security import (
    SecurityLayer,
    SecurityParameters,
    UMBRAL_QBER_SEGURIDAD,
)

# ============================================================================
# 1. Simulador Principal (Facade de Orquestación)
# ============================================================================

class BB84Simulator:
    """
    Orquestador principal desacoplado que ejecuta la simulación de 3 capas.

    Hipótesis del Modelo:
    El protocolo asume la existencia de un canal clásico autenticado exógeno.
    La fase de confirmación de clave (confirmacion_clave_ok) comprueba la coincidencia
    de las claves resultantes (mediante hashes/tags), pero no autentica la identidad
    de las partes frente a ataques de tipo Man-In-The-Middle (MITM) en el canal clásico.
    """

    def __init__(
        self,
        entropy: EntropySource,
        sec_params: SecurityParameters | None = None,
    ):
        # Inyección de dependencias para la generación de números aleatorios
        self.entropy = entropy
        # Si no se proveen parámetros de seguridad explícitos, usamos los valores por defecto
        self.sec_params = sec_params or SecurityParameters()
        
        # Inicialización de las capas clásicas y de seguridad
        # ClassicalLayer maneja sifting, estimación de parámetros, Cascade y Toeplitz
        self.classical_layer = ClassicalLayer(self.entropy, self.sec_params)
        # SecurityLayer evalúa cotas teóricas (Serfling, LHL) y construye el reporte final
        self.security_layer = SecurityLayer(self.sec_params)

    # Defaults de los parámetros propios de FiberChannel, usados por run()
    # tanto para construir el canal por defecto como para detectar cuándo
    # se ha pasado un `channel` explícito JUNTO a alguno de ellos con un
    # valor distinto del suyo (ver docstring de run()).
    _FIBER_DEFAULTS: ClassVar[dict[str, Any]] = {
        "distancia_km": 0.0,
        "atenuacion_db_km": 0.2,
        "eta_detector": 0.2,
        "prob_dark_count": 1e-6,
        "qber_intrinseco": 0.01,
        "eve": None,
        "eve_fraction": 1.0,
    }

    def run(
        self,
        n_qubits: int = 20_000,
        distancia_km: float = 0.0,
        atenuacion_db_km: float = 0.2,
        eta_detector: float = 0.2,
        prob_dark_count: float = 1e-6,
        qber_intrinseco: float = 0.01,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
        fraccion_verificacion: float = 0.15,
        n_min_tamizados: int = 100,
        n_pasadas_cascade: int = 4,
        channel: ChannelModel | None = None,
    ) -> SecurityReport:
        """
        v5.4: `channel` permite inyectar un ChannelModel que no sea de
        fibra (DepolarizingChannel, FreeSpaceChannel, o uno propio) sin
        tener que orquestar QuantumLayer + ClassicalLayer + SecurityLayer
        a mano, como exigía v5.3 (ver el bloque `if __name__ ==
        "__main__":` de versiones anteriores).

        Si `channel` es None (comportamiento por defecto, igual que en
        versiones previas), se construye un FiberChannel con
        distancia_km/atenuacion_db_km/eta_detector/prob_dark_count/
        qber_intrinseco/eve/eve_fraction.

        Si se proporciona `channel`, esos siete parámetros dejan de
        tener efecto sobre el canal (el que se usa es el que se pasó).
        Para que esto no falle en SILENCIO -- alguien podría creer que
        está configurando el canal inyectado a través de ellos -- run()
        exige que todos permanezcan en su valor por defecto cuando
        `channel` no es None; si alguno se pasó explícitamente con otro
        valor, lanza ValueError en vez de ignorarlo.

        `SecurityReport.distancia_km` se toma de `channel.distancia_km`
        si el canal expone ese atributo (como FiberChannel) y queda en
        None en caso contrario (DepolarizingChannel y FreeSpaceChannel
        no tienen una noción de distancia de fibra).
        """
        # Validaciones exhaustivas de parámetros enteros y de probabilidad/distancia
        validar_entero_positivo("n_qubits", n_qubits)
        validar_entero_positivo("n_min_tamizados", n_min_tamizados)
        validar_entero_positivo("n_pasadas_cascade", n_pasadas_cascade)
        validar_probabilidad("fraccion_verificacion", fraccion_verificacion)
        validar_probabilidad("eta_detector", eta_detector)
        validar_probabilidad("prob_dark_count", prob_dark_count)
        validar_probabilidad("qber_intrinseco", qber_intrinseco)
        validar_probabilidad("eve_fraction", eve_fraction)
        validar_no_negativo("distancia_km", distancia_km)
        validar_no_negativo("atenuacion_db_km", atenuacion_db_km)
        # Agrupamos los parámetros físicos por defecto para facilitar la validación
        valores_fibra = {
            "distancia_km": distancia_km,
            "atenuacion_db_km": atenuacion_db_km,
            "eta_detector": eta_detector,
            "prob_dark_count": prob_dark_count,
            "qber_intrinseco": qber_intrinseco,
            "eve": eve,
            "eve_fraction": eve_fraction,
        }

        # Lógica de inyección de dependencias para el canal
        if channel is not None:
            # Comprobamos que no se intenten sobreescribir variables de fibra si se pasa un canal custom
            no_default = [
                nombre for nombre, valor in valores_fibra.items()
                if valor != self._FIBER_DEFAULTS[nombre]
            ]
            if no_default:
                raise ValueError(
                    "Se ha proporcionado 'channel' junto con parámetros "
                    f"propios de FiberChannel que se ignorarían en "
                    f"silencio: {sorted(no_default)}. Pásalos directamente "
                    "al construir ese ChannelModel, o no proporciones "
                    "'channel' para usar el FiberChannel por defecto."
                )
        else:
            # Construcción del canal de fibra óptica por defecto
            channel = FiberChannel(entropy=self.entropy, **valores_fibra)

        # Extracción segura de la distancia (retorna None si el canal no tiene este atributo)
        distancia_km_reporte = getattr(channel, "distancia_km", None)

        # ==========================================
        # FASE 1: Transmisión Cuántica
        # ==========================================
        
        # Instanciamos los actores principales
        alice = Alice(self.entropy)
        bob = Bob(self.entropy)
        quantum_layer = QuantumLayer(alice, bob, channel)

        # Ejecutamos la simulación física (preparación, canal con ruido/Eve, y medición de Bob)
        detection = quantum_layer.execute(n_qubits)

        # ==========================================
        # FASE 2: Post-procesamiento Clásico
        # ==========================================
        
        # Sifting (Tamizado): Descartamos los bits donde Alice y Bob usaron bases distintas
        clave_alice, clave_bob = self.classical_layer.sifting(detection)
        n_tamizada = len(clave_alice)

        # Salida temprana: Si la atenuación fue demasiada, no hay suficientes bits para continuar
        if n_tamizada < n_min_tamizados:
            return self.security_layer.evaluate_and_build(
                detection, n_tamizada, None, 0, 0, False,
                np.array([]), np.array([]), distancia_km_reporte,
                abort_reason="Insuficientes bits tamizados."
            )

        # Estimación de Parámetros (Parameter Estimation - PE)
        # Sacrificamos una fracción aleatoria de la clave tamizada para estimar el QBER
        pe_data = self.classical_layer.parameter_estimation(
            clave_alice, clave_bob, fraccion_verificacion
        )

        # Salida temprana: Si la cota de error supera el umbral, asumimos presencia de Eve
        # Esto evita consumir recursos computacionales en Cascade y Toeplitz innecesariamente.
        if pe_data["phase_error_bound"].value >= UMBRAL_QBER_SEGURIDAD:
            return self.security_layer.evaluate_and_build(
                detection, n_tamizada, pe_data, 0, 0, False,
                np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), distancia_km_reporte,
                abort_reason=(
                    f"QBER Cota ({pe_data['phase_error_bound'].value:.4f}) "
                    f"supera umbral ({UMBRAL_QBER_SEGURIDAD}) -> posible espía."
                ),
            )

        # Claves restantes tras sacrificar los bits para la estimación
        alice_resto = pe_data["clave_alice_resto"]
        bob_resto = pe_data["clave_bob_resto"]

        # Corrección de Errores (Error Correction - Cascade)
        # Bob reconcilia su clave con la de Alice intercambiando paridades públicamente
        bob_reconciliado, leak_ec, discrepancias = (
            self.classical_layer.error_correction_cascade(
                alice_resto,
                bob_resto,
                qber_estimado=pe_data["bit_error"].value,
                n_pasadas=n_pasadas_cascade,
            )
        )

        # Confirmación de Claves (Key Confirmation)
        # Se verifica mediante un hash si ambas claves son idénticas tras Cascade
        confirmacion_clave_ok = self.classical_layer.key_confirmation(alice_resto, bob_reconciliado)

        # SALIDA TEMPRANA: Si las claves no coinciden tras Cascade, abortamos
        # evitando el cálculo innecesario de la matriz de Toeplitz por FFT.
        if not confirmacion_clave_ok:
            return self.security_layer.evaluate_and_build(
                detection,
                n_tamizada,
                pe_data,
                leak_ec,
                discrepancias,
                False,
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.uint8),
                distancia_km_reporte,
                abort_reason="Fallo en la confirmación de clave: existen discrepancias no corregidas tras Cascade.",
            )

        # Amplificación de Privacidad (Privacy Amplification - LHL)
        # 1. Calculamos la longitud de clave segura considerando la información filtrada a Eve (leak_ec y cota Serfling)
        target_len = self.security_layer.calculate_lhl_length(
            n_resto=len(alice_resto),
            e_ph=pe_data["phase_error_bound"],
            leak_ec=leak_ec,
            tag_length=self.sec_params.tag_length_efectivo,
        )

        # 2. Comprimimos la clave usando una matriz de Toeplitz compartida (generada vía semilla pública)
        seed_pa = self.entropy.random_seed_int()
        clave_alice_pa = self.classical_layer.privacy_amplification_toeplitz(
            alice_resto, target_len, seed_pa
        )
        clave_bob_pa = self.classical_layer.privacy_amplification_toeplitz(
            bob_reconciliado, target_len, seed_pa
        )

        # ==========================================
        # FASE 3: Capa de Seguridad y Reporte
        # ==========================================
        
        # Generación del informe final con métricas consolidadas
        return self.security_layer.evaluate_and_build(
            detection=detection,
            n_tamizada=n_tamizada,
            pe_data=pe_data,
            bits_revelados_ec=leak_ec,
            discrepancias=discrepancias,
            confirmacion_clave_ok=confirmacion_clave_ok,
            clave_alice_pa=clave_alice_pa,
            clave_bob_pa=clave_bob_pa,
            distancia_km=distancia_km_reporte,
        )


# ============================================================================
# 2. Suite de Tests Estadísticos (Aseguramiento de Calidad)
# ============================================================================

def run_statistical_tests():
    """
    Ejecuta verificaciones estadísticas automáticas del simulador.

    Esta revisión amplía los 3 tests que ya traía (sifting, QBER de Eve, atenuación)
    con: uniformidad de bits/bases de Alice, tasa de dark counts, y un
    test de regresión específico para Cascade. Este último existe
    porque, durante la revisión que dio lugar a esta versión, se
    encontró (en OTRA implementación de Cascade con backtracking, no en
    esta) un bug en el que las paridades de pasadas futuras se
    calculaban una sola vez y nunca se refrescaban con las correcciones
    ya aplicadas por pasadas anteriores -- dejaba decenas de
    discrepancias sin corregir a partir de QBER~3%, y su propio test
    unitario (con muy pocos errores y una única semilla) no lo detectó.
    La lección: un test de Cascade con 3-6 errores fijos en una clave de
    juguete no tiene potencia estadística para detectar ese tipo de
    fallo; hay que barrer QBER realista sobre muchas realizaciones.
    """
    print("\n=== Ejecutando Tests Estadísticos de Validación ===")
    
    # Fijamos la semilla para asegurar la reproducibilidad de los tests
    entropy = EntropySource.simulation(seed=42)
    sec_params = SecurityParameters()
    sim = BB84Simulator(entropy, sec_params)

    # Test 1: Sifting (~50%)
    # Verifica que estadísticamente, la mitad de las bases coincidan.
    res_ideal = sim.run(n_qubits=100_000, qber_intrinseco=0.0, eta_detector=1.0)
    ratio_sift = res_ideal.n_tamizada / 100_000
    assert 0.48 < ratio_sift < 0.52, f"Falló Sifting Ratio: {ratio_sift}"
    print(f" [PASS] Sifting Rate Ratio ideal: {ratio_sift:.4f} (~0.50)")

    # Test 2: QBER con Intercept-Resend completo (debe dar ~25%)
    # Un ataque IR completo genera un QBER teórico del 25% tras el sifting.
    eve_full = InterceptResendEve()
    res_eve = sim.run(
        n_qubits=100_000,
        qber_intrinseco=0.0,
        eta_detector=1.0,
        eve=eve_full,
        eve_fraction=1.0,
    )
    qber_eve = res_eve.qber_medido
    assert 0.23 < qber_eve < 0.27, f"Falló QBER de Eve: {qber_eve}"
    print(f" [PASS] Intercept-Resend QBER: {qber_eve:.4f} (~0.25)")

    # Test 3: Pérdidas del Canal (Atenuación)
    # Verifica que la ecuación de transmitancia η = 10^(-(α * L) / 10) se cumpla empíricamente.
    res_canal = sim.run(
        n_qubits=100_000,
        distancia_km=10.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=0.0,
    )
    eta_teorica = 10 ** (-0.2 * 10 / 10)  # ~0.6309
    eta_medida = res_canal.detector_click_rate
    assert math.isclose(eta_medida, eta_teorica, rel_tol=0.05)
    print(f" [PASS] Canal Attenuation Click Rate: {eta_medida:.4f} (Teórico: {eta_teorica:.4f})")

    # Test 4: Uniformidad de bits y bases de Alice
    # Se evalúa la media y el test Chi-cuadrado para asegurar la calidad de la fuente entrópica.
    alice_test = Alice(EntropySource.simulation(seed=123))
    n_unif = 200_000
    paquete = alice_test.prepare(n_unif)
    
    media_bits = float(np.mean(paquete.bits))
    media_bases = float(np.mean(paquete.bases))
    assert 0.49 < media_bits < 0.51, f"Falló uniformidad de bits: {media_bits}"
    assert 0.49 < media_bases < 0.51, f"Falló uniformidad de bases: {media_bases}"
    print(f" [PASS] Uniformidad de bits de Alice: {media_bits:.4f} (~0.50)")
    print(f" [PASS] Uniformidad de bases de Alice: {media_bases:.4f} (~0.50)")

    # Validación Chi-cuadrado para distribución uniforme de bits
    conteo_bits = np.bincount(paquete.bits, minlength=2)
    chi2_bits = float(np.sum((conteo_bits - n_unif / 2) ** 2) / (n_unif / 2))
    # Umbral 10.83 para p<0.001 con 1 grado de libertad
    assert chi2_bits < 10.83, f"Chi-cuadrado de bits sospechosamente alto: {chi2_bits:.3f}"
    print(f" [PASS] Chi-cuadrado uniformidad de bits: {chi2_bits:.3f} (< 10.83 a p<0.001)")

    # Validación Chi-cuadrado para distribución uniforme de bases
    conteo_bases = np.bincount(paquete.bases, minlength=2)
    chi2_bases = float(np.sum((conteo_bases - n_unif / 2) ** 2) / (n_unif / 2))
    assert chi2_bases < 10.83, f"Chi-cuadrado de bases sospechosamente alto: {chi2_bases:.3f}"
    print(f" [PASS] Chi-cuadrado uniformidad de bases: {chi2_bases:.3f} (< 10.83 a p<0.001)")

    # Test de contingencia 2x2 para descartar anti-correlaciones entre bits y bases
    tabla = np.zeros((2, 2), dtype=int)
    np.add.at(tabla, (paquete.bits, paquete.bases), 1)
    esperado = (
        tabla.sum(axis=1, keepdims=True)
        * tabla.sum(axis=0, keepdims=True)
        / tabla.sum()
    )
    chi2_independencia = float(np.sum((tabla - esperado) ** 2 / esperado))
    assert chi2_independencia < 10.83, (
        f"Bits y bases no parecen independientes: chi²={chi2_independencia:.3f}"
    )
    print(f" [PASS] Chi-cuadrado independencia bit-base: {chi2_independencia:.3f} (< 10.83 a p<0.001)")

    # Test 5: Tasa de dark counts
    # Saturamos el canal a 500 km, por lo que casi todo click debe ser un dark count.
    res_dark = sim.run(
        n_qubits=200_000,
        distancia_km=500.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=1e-3,
        qber_intrinseco=0.0,
    )
    assert math.isclose(res_dark.detector_click_rate, 2e-3, rel_tol=0.25)
    print(f" [PASS] Dark count rate a canal saturado: {res_dark.detector_click_rate:.5f} (objetivo ~0.00200)")
    
    # Test 6: Regresión de Error Correction (Cascade)
    # Se inyectan fallos de forma controlada y se verifica que Cascade resuelva el 100%.
    rng_test = np.random.default_rng(2026)
    n_test = 20_000
    for qber_test in [0.01, 0.03, 0.05, 0.08]:
        alice_k = rng_test.integers(0, 2, size=n_test)
        bob_k = alice_k.copy()
        
        # Inyección de errores basados en QBER
        flips = rng_test.random(n_test) < qber_test
        bob_k[flips] ^= 1
        
        capa_test = ClassicalLayer(
            EntropySource.simulation(int(rng_test.integers(0, 2**31))), sec_params
        )
        # Ejecutamos la reconciliación
        _, _, discrepancias = capa_test.error_correction_cascade(
            alice_k, bob_k, qber_estimado=qber_test, n_pasadas=4
        )
        assert discrepancias == 0, (
            f"Cascade dejó {discrepancias} discrepancias sin corregir a "
            f"QBER={qber_test:.1%} (n={n_test})"
        )
    print(" [PASS] Cascade corrige el 100% de los errores inyectados (QBER 1%-8%, n=20 000)")

    print("=== Todos los tests pasaron correctamente. ===\n")