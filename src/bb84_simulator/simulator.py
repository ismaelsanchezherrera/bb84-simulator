"""
Simulador BB84 (Distribución de Claves Cuánticas).

Módulo principal para la simulación del protocolo QKD BB84, incluyendo
modelado de canales cuánticos con ruido, detección de intromisión (Eve),
reconciliación de información por Cascade y amplificación de privacidad.

"""

from __future__ import annotations

from bb84_simulator.entropy import EntropySource
from bb84_simulator.models import DetectionResult, SecurityReport
from bb84_simulator.validation import (
    validar_entero_positivo,
    validar_epsilon as _validar_epsilon,
    validar_no_negativo as _validar_no_negativo,
    validar_probabilidad as _validar_prob01,
)
from bb84_simulator.attacks import (
    CollectiveAttack,
    EveStrategy,
    InterceptResendEve,
    PassiveEve,
    QuantumPacket,
    RandomEve,
)
from bb84_simulator.channels import (
    ChannelModel,
    DepolarizingChannel,
    FiberChannel,
    FreeSpaceChannel,
)

from bb84_simulator.quantum import Alice, Bob, QuantumLayer

import math
import os
import secrets
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np

# ============================================================================
# 0. Parámetros de Seguridad y Teoría de la Información
# ============================================================================

UMBRAL_QBER_SEGURIDAD = 0.11

@dataclass(frozen=True)
class SecurityParameters:
    """
    Parámetros de seguridad unificados según el marco composable moderno.

    `tag_length` es opcional: si no se fija explícitamente, se DERIVA de `epsilon_auth` mediante
    `tag_length_efectivo`, de modo que ambos queden siempre consistentes
    (ver `ClassicalLayer.key_confirmation`).

    `epsilon_ec` sigue sin tener una fórmula analítica cerrada para
    Cascade (a diferencia de epsilon_pe/epsilon_pa, que sí la tienen vía
    Serfling y Leftover Hash Lemma respectivamente) — se documenta como
    objetivo de diseño; la fracción de ejecuciones con
    `discrepancias_tras_cascade > 0` observada empíricamente es la forma
    correcta de verificarlo en este simulador.
    """
    epsilon_pe: float = 1e-10       # Confianza de la cota de Serfling (1974)
    epsilon_pa: float = 1e-10       # Término de suavizado del Leftover Hash Lemma
    epsilon_auth: float = 1e-12     # Prob. de colisión aceptada en key confirmation
    epsilon_ec: float = 1e-10       # Objetivo de fallo de reconciliación (de diseño)
    fec_efficiency: float = 1.10    # f_EC de referencia; Cascade real ~1.05-1.20
    tag_length: int | None = None  # Si es None, se deriva de epsilon_auth

    def __post_init__(self) -> None:
        """v5.4: sin esta validación, p.ej. epsilon_pa=2 se aceptaba y
        volvía negativo (por tanto indebidamente favorable) el término
        2*log2(1/epsilon_pa) de calculate_lhl_length -- un fallo que
        infla la longitud de clave declarada segura. Los cuatro epsilons son términos de
        confianza y deben estar en (0, 1) abierto; fec_efficiency es un
        factor multiplicativo sobre una fuga real y debe ser >= 1;
        tag_length, si se fija a mano, sustituye por completo la
        derivación desde epsilon_auth (ver tag_length_efectivo) y debe
        ser un entero positivo."""
        for nombre in ("epsilon_pe", "epsilon_pa", "epsilon_auth", "epsilon_ec"):
            _validar_epsilon(nombre, getattr(self, nombre))

        if (
            isinstance(self.fec_efficiency, bool)
            or not isinstance(self.fec_efficiency, Real)
            or not math.isfinite(self.fec_efficiency)
            or self.fec_efficiency < 1.0
        ):
            raise ValueError("fec_efficiency debe ser un real finito >= 1.")

        if self.tag_length is not None and (
            isinstance(self.tag_length, bool)
            or not isinstance(self.tag_length, int)
            or self.tag_length <= 0
        ):
            raise ValueError("tag_length debe ser un entero positivo o None.")

    @property
    def tag_length_efectivo(self) -> int:
        """Longitud de tag que garantiza epsilon_auth de prob. de colisión
        para un hash universal (familia Toeplitz, Carter & Wegman 1979),
        salvo que el usuario haya fijado `tag_length` explícitamente."""
        if self.tag_length is not None:
            return self.tag_length
        return max(1, math.ceil(math.log2(1.0 / self.epsilon_auth)))


@dataclass(frozen=True)
class BitErrorEstimate:
    """
    Estimación del error de BIT (QBER), Serfling (1974) para la cota.

    value: estimación puntual del QBER sobre la submuestra de
    verificación (no confundir con la cota superior, ver
    PhaseErrorEstimate).
    """
    value: float
    n_muestra: int = 0
    n_poblacion: int = 0
    epsilon: float = 0.0


@dataclass(frozen=True)
class PhaseErrorEstimate:
    """
    Cota superior del error de FASE, e_ph: la cantidad que entra
    realmente en la prueba de seguridad de Shor & Preskill (2000) vía
    reducción a un código CSS.

    LIMITACIÓN DOCUMENTADA: aquí
    e_ph se toma igual a la cota de Serfling del error de BIT
    (supuesto de canal simétrico/depolarizante). Para un adversario
    general e_ph podría estimarse de forma independiente; se mantiene
    esta clase separada de BitErrorEstimate precisamente para que ese
    cambio, no obligue a tocar el resto del pipeline.
    """
    value: float
    supuesto: str = "e_ph := cota_serfling(qber_bit); canal simétrico, no derivado de forma independiente"


def entropia_binaria(x: float | np.ndarray) -> float | np.ndarray:
    """Entropía binaria h2(x), con h2(0)=h2(1)=0."""
    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(x_arr)
    mask = (x_arr > 0.0) & (x_arr < 1.0)
    out[mask] = (
        -x_arr[mask] * np.log2(x_arr[mask])
        - (1.0 - x_arr[mask]) * np.log2(1.0 - x_arr[mask])
    )
    if np.ndim(x) == 0:
        return float(out)
    return out


def cota_serfling_superior(
    q_estimado: float,
    n_muestra: int,
    n_poblacion: int,
    epsilon: float = 1e-10,
) -> float:
    """
    Cota superior, con confianza 1-epsilon, de la tasa de error del RESTO
    (los n_poblacion - n_muestra bits NO muestreados), a partir de la tasa
    observada en la submuestra. Muestreo sin reemplazo, Serfling (1974).

    CORRECCIÓN v5.3 (fallo de contabilidad a nivel de prueba, misma familia
    que los de la cabecera de v5.2): la versión anterior devolvía la cota de
    Serfling sobre la media de la POBLACIÓN completa. Pero la cantidad que
    entra en la prueba de seguridad (Shor-Preskill vía e_ph, y el término
    h2(e_ph) del LHL) es la tasa de error de los bits QUE FORMAN LA CLAVE,
    es decir, del resto no muestreado. Con E errores totales (adversariales,
    fijados por Eve) la identidad exacta

        q_resto = (E - n*q_est) / (N - n)

    implica  {q_resto > q_est + m}  <=>  {q_pob - q_est > m*(N-n)/N},
    así que usar el margen de población m directamente sobre el resto solo
    garantiza un epsilon EFECTIVO de  eps^((N-n)/N)^2  (p.ej. con
    fraccion_verificacion=0.15 y eps=1e-10: ~6e-8, unas 600 veces peor que
    lo declarado). La corrección es escalar el margen por N/(N-n), que es
    la forma en que la literatura finite-key aplica Serfling al resto
    (cf. Fung, Ma & Chau 2010; Tomamichel, Lim, Gisin & Renner 2012).

    Nota empírica honesta: la holgura intrínseca de Serfling frente a la
    cola hipergeométrica exacta hace que la versión sin corregir rara vez
    se viole en la práctica a estos parámetros; el fallo es de la GARANTÍA
    FORMAL declarada, no una brecha observable en Monte Carlo. Coste de la
    corrección: el margen crece un factor 1/(1-fraccion_verificacion)
    (~18% con la fracción por defecto).
    """
    if n_muestra <= 0 or n_poblacion <= 0:
        return 1.0
    if n_muestra >= n_poblacion:
        return float(np.clip(q_estimado, 0.0, 1.0))
    if not 0 < epsilon < 1:
        raise ValueError("epsilon debe estar entre 0 y 1.")

    factor = (n_poblacion - n_muestra + 1) / n_poblacion
    margen_poblacion = np.sqrt(factor * np.log(1.0 / epsilon) / (2.0 * n_muestra))
    # Escalado poblacion -> resto no muestreado (ver docstring).
    margen_resto = margen_poblacion * n_poblacion / (n_poblacion - n_muestra)
    return float(min(1.0, q_estimado + margen_resto))

# ============================================================================
# 4. Capa Clásica (Classical Layer)
# ============================================================================

def _paridad(bits: np.ndarray) -> int:
    return int(np.sum(bits) & 1)


def _convolucion_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_total = len(a) + len(b) - 1
    n_fft = 1 << (n_total - 1).bit_length()

    A = np.fft.rfft(a, n_fft)
    B = np.fft.rfft(b, n_fft)
    conv = np.fft.irfft(A * B, n_fft)[:n_total]

    redondeo = float(np.max(np.abs(conv - np.round(conv))))
    if redondeo > 1e-3:
        raise RuntimeError(f"Redondeo FFT sospechoso: {redondeo:.2e}")

    return np.round(conv).astype(np.int64)


class ClassicalLayer:
    """Capa 2: Responsable del post-procesamiento clásico de información."""

    def __init__(self, entropy: EntropySource, sec_params: SecurityParameters):
        self.entropy = entropy
        self.sec_params = sec_params

    def sifting(self, detection: DetectionResult) -> tuple[np.ndarray, np.ndarray]:
        mask = detection.bases_alice == detection.bases_bob
        return detection.bits_alice[mask].copy(), detection.bits_bob[mask].copy()

    def parameter_estimation(
        self,
        clave_alice: np.ndarray,
        clave_bob: np.ndarray,
        fraccion_verificacion: float = 0.15,
    ) -> dict[str, Any]:
        if not 0.0 < fraccion_verificacion < 1.0:
            raise ValueError("fraccion_verificacion debe estar en el intervalo (0, 1).")

        n = len(clave_alice)
        if n == 0:
            raise ValueError("No hay bits tamizados para estimación.")

        n_verif = max(1, int(n * fraccion_verificacion))
        idx_verif = self.entropy.choice(np.arange(n), size=n_verif, replace=False)

        errores = clave_alice[idx_verif] != clave_bob[idx_verif]
        qber_puntual = float(np.mean(errores))

        qber_superior = cota_serfling_superior(
            qber_puntual, n_verif, n, self.sec_params.epsilon_pe
        )

        mascara_resto = np.ones(n, dtype=bool)
        mascara_resto[idx_verif] = False

        return {
            "bit_error": BitErrorEstimate(
                qber_puntual, n_muestra=n_verif, n_poblacion=n,
                epsilon=self.sec_params.epsilon_pe,
            ),
            "phase_error_bound": PhaseErrorEstimate(qber_superior),
            "n_verificacion": n_verif,
            "clave_alice_resto": clave_alice[mascara_resto],
            "clave_bob_resto": clave_bob[mascara_resto],
        }

    def error_correction_cascade(
        self,
        clave_alice: np.ndarray,
        clave_bob: np.ndarray,
        qber_estimado: float,
        n_pasadas: int = 4,
    ) -> tuple[np.ndarray, int, int]:
        """
        Cascade con backtracking en cola global (Brassard & Salvail,
        1993, "Secret-Key Reconciliation by Public Discussion",
        EUROCRYPT '93). El tamaño de bloque inicial ~0.73/QBER y su
        duplicación en cada pasada siguen la heurística original del
        artículo.

        Backtracking: `bloques_de_posicion[i]` precalcula, para cada
        índice ORIGINAL, en qué (pasada, bloque) participa a lo largo de
        TODAS las pasadas. Al corregir un bit se refresca de inmediato
        la paridad de todos esos bloques -- incluidos los de pasadas
        futuras, ya construidas pero aún no procesadas -- y se reencola
        cualquiera que pase a discordar. Esto es lo que evita el fallo
        de diseño más frecuente en implementaciones de Cascade con
        backtracking: calcular las paridades de todas las pasadas una
        sola vez al principio y no refrescarlas nunca con las
        correcciones ya aplicadas (verificado empíricamente que ESTA
        implementación no lo sufre: ver test de regresión en
        run_statistical_tests).

        v5.4: se valida n_pasadas >= 1, que ambas claves tengan la
        misma longitud, y que qber_estimado esté en [0, 0.5]. Nota
        importante sobre n_pasadas=0: NO deja, en la práctica, un
        "éxito falso" silencioso -- ya existía el guardado
        `q = max(qber_estimado, 1/n)` (nunca hay división por 0), y si
        Alice y Bob quedan con claves distintas, key_confirmation() lo
        detecta y run() aborta el informe (salvo colisión de hash,
        acotada por epsilon_auth). Aun así se rechaza aquí: evita
        cómputo inútil, un resultado confuso, y no debería depender de
        que otra capa lo atrape para ser correcto.

        v5.4.1: n_pasadas debe ser además un ENTERO (y no bool, que es
        subclase de int). `n_pasadas < 1` por sí solo dejaba pasar
        n_pasadas=1.5 (1.5 >= 1, no viola esa cota), que más abajo
        rompía con un TypeError de numpy/range al intentar usarlo como
        índice o tamaño -- un fallo interno, no un ValueError con
        mensaje claro sobre qué se pasó mal.
        """
        if (
            isinstance(n_pasadas, bool)
            or not isinstance(n_pasadas, (int, np.integer))
            or n_pasadas < 1
        ):
            raise ValueError("n_pasadas debe ser un entero >= 1.")

        if len(clave_alice) != len(clave_bob):
            raise ValueError("Las claves de Alice y Bob deben tener la misma longitud.")

        if not 0.0 <= qber_estimado <= 0.5:
            raise ValueError("qber_estimado debe estar entre 0 y 0.5.")

        n = len(clave_alice)
        if n == 0:
            return clave_bob.copy(), 0, 0

        q = max(float(qber_estimado), 1.0 / n)
        q = min(q, 0.5)
        block_size = max(2, min(n, int(np.ceil(0.73 / q))))

        bob_actual = clave_bob.copy()
        bits_revelados = 0

        # Las permutaciones de Cascade son PÚBLICAS (se anuncian por el
        # canal clásico autenticado para que Alice y Bob las repliquen
        # de forma idéntica); no necesitan salir de self.entropy en su
        # modo actual (simulation/crypto). Se saca UNA semilla de
        # self.entropy y se expande con un generador determinista y
        # rápido -- el mismo patrón que ya usa correctamente
        # privacy_amplification_toeplitz. Efecto práctico: en modo
        # crypto, deja de pagarse el coste de 4 barajados con el CSPRNG
        # del sistema operativo en cada ejecución.
        semilla_publica_cascade = self.entropy.random_seed_int()
        rng_publico = np.random.default_rng(semilla_publica_cascade)

        bloques_por_pasada = []
        # bloque_de_idx[pasada][i] = id del bloque de la pasada `pasada` que
        # contiene el índice ORIGINAL i. Sustituye (v5.3) a la antigua lista
        # de listas `bloques_de_posicion`, que se construía con un bucle
        # Python de n*n_pasadas appends con conversión int() por elemento.
        # Se calcula vectorizado vía la permutación inversa: la posición de
        # i dentro de perm es inv_perm[i], y su bloque es inv_perm[i] // bs.
        bloque_de_idx: list[np.ndarray] = []

        for pasada in range(n_pasadas):
            perm = rng_publico.permutation(n)
            inv_perm = np.empty(n, dtype=np.int64)
            inv_perm[perm] = np.arange(n, dtype=np.int64)
            bloque_de_idx.append(inv_perm // block_size)

            bloques = []
            for inicio in range(0, n, block_size):
                indices_perm = np.arange(
                    inicio, min(inicio + block_size, n), dtype=np.int64
                )
                indices_originales = perm[indices_perm]
                bloques.append({
                    "perm": perm,
                    "indices_perm": indices_perm,
                    "indices_originales": indices_originales,
                    "pasada": pasada,
                    "id": len(bloques),
                })

            bloques_por_pasada.append(bloques)
            block_size = min(n, block_size * 2)

        # v5.3: paridades iniciales por bloque vectorizadas con
        # np.add.reduceat sobre la clave permutada (antes: bucle Python de
        # _paridad + fancy indexing bloque a bloque; en la primera pasada
        # hay ~n*QBER/0.73 bloques, que a QBER bajo son miles).
        paridades_alice: list[list[int]] = []
        paridades_bob: list[list[int]] = []
        for blqs in bloques_por_pasada:
            perm_p = blqs[0]["perm"]
            inicios = np.fromiter(
                (int(b["indices_perm"][0]) for b in blqs), dtype=np.int64
            )
            paridades_alice.append(
                (np.add.reduceat(clave_alice[perm_p], inicios) & 1).tolist()
            )
            paridades_bob.append(
                (np.add.reduceat(bob_actual[perm_p], inicios) & 1).tolist()
            )

        # Anunciar la paridad de un bloque cuesta 1 bit de comunicación
        # por el canal clásico SE DETECTE O NO discrepancia (la versión recibida solo
        # contaba los bits gastados en la búsqueda binaria de los
        # bloques que sí discrepaban al principio, subestimando la fuga
        # real hacia Eve -- ver leak_ec_real en evaluate_and_build).
        bits_revelados += sum(len(blqs) for blqs in bloques_por_pasada)

        cola = deque()
        en_cola = set()

        def enqueue(pasada: int, bloque_id: int):
            key = (pasada, bloque_id)
            if key not in en_cola:
                cola.append(key)
                en_cola.add(key)

        for p, blqs in enumerate(bloques_por_pasada):
            for bid in range(len(blqs)):
                if paridades_alice[p][bid] != paridades_bob[p][bid]:
                    enqueue(p, bid)

        while cola:
            pasada, bloque_id = cola.popleft()
            en_cola.discard((pasada, bloque_id))

            # v5.3: las paridades almacenadas son EXACTAS en todo momento
            # (cada corrección de un bit voltea, vía XOR, la paridad de
            # todos los bloques que lo contienen -- ver el bucle de
            # backtracking más abajo), así que no hace falta recomputar
            # _paridad sobre el bloque completo al desencolar. En pasadas
            # tardías los bloques son enormes (block_size se duplica cada
            # pasada) y ese recómputo dominaba el coste del backtracking.
            if paridades_alice[pasada][bloque_id] == paridades_bob[pasada][bloque_id]:
                continue

            bloque = bloques_por_pasada[pasada][bloque_id]

            perm = bloque["perm"]
            indices_perm = bloque["indices_perm"]
            idx_p_curr = indices_perm.copy()

            while len(idx_p_curr) > 1:
                mitad = len(idx_p_curr) // 2
                izq, der = idx_p_curr[:mitad], idx_p_curr[mitad:]
                bits_revelados += 1
                if _paridad(clave_alice[perm[izq]]) != _paridad(bob_actual[perm[izq]]):
                    idx_p_curr = izq
                else:
                    idx_p_curr = der

            bits_revelados += 1
            err_orig = int(perm[idx_p_curr[0]])
            bob_actual[err_orig] ^= 1

            # v5.3: volteo de un solo bit => la paridad de CADA bloque que
            # lo contiene se invierte, determinísticamente. Actualización
            # O(n_pasadas) por corrección en vez de O(n_pasadas *
            # block_size) recomputando _paridad de bloques completos.
            for p_af in range(n_pasadas):
                b_af = int(bloque_de_idx[p_af][err_orig])
                nueva = paridades_bob[p_af][b_af] ^ 1
                paridades_bob[p_af][b_af] = nueva
                if nueva != paridades_alice[p_af][b_af]:
                    enqueue(p_af, b_af)

        discrepancias = int(np.count_nonzero(clave_alice != bob_actual))
        return bob_actual, bits_revelados, discrepancias

    def privacy_amplification_toeplitz(
        self,
        clave: np.ndarray,
        longitud_salida: int,
        semilla_publica: int | None = None,
    ) -> np.ndarray:
        """
        Amplificación de privacidad mediante hash universal mediante
        matriz de Toeplitz binaria (familia 2-universal, Carter & Wegman
        1979, "Universal Classes of Hash Functions"; aplicación a QKD
        en Bennett, Brassard, Robert 1988, "Privacy Amplification by
        Public Discussion"). La multiplicación matriz-vector se calcula
        como convolución vía FFT (_convolucion_fft): O(N log N) en vez
        de O(N * longitud_salida).

        `semilla_publica` NO necesita ser secreta (el Leftover Hash
        Lemma solo exige que se elija con independencia de la
        información de Eve, no que se oculte); por eso su expansión usa
        un PRNG determinista estándar (`np.random.default_rng`), no
        EntropySource -- ver la nota de diseño en la clase EntropySource.
        """
        longitud_salida = max(0, int(longitud_salida))
        n = len(clave)
        if longitud_salida == 0 or n == 0:
            return np.array([], dtype=np.uint8)

        if semilla_publica is None:
            semilla_publica = self.entropy.random_seed_int()

        rng = np.random.default_rng(semilla_publica)
        semilla_toeplitz = rng.integers(0, 2, size=longitud_salida + n - 1)

        conv = _convolucion_fft(semilla_toeplitz.astype(np.int64), clave.astype(np.int64))
        y = conv[n - 1 : n - 1 + longitud_salida]
        return (y & 1).astype(np.uint8)

    def key_confirmation(self, clave_a: np.ndarray, clave_b: np.ndarray) -> bool:
        """
        Confirmación de igualdad de claves mediante hash universal
        (familia Toeplitz, Carter & Wegman 1979). Un tag de
        `tag_length_efectivo` bits da una probabilidad de colisión
        (dos claves DISTINTAS con el mismo tag) de ~2^-tag_length, que
        es exactamente `epsilon_auth` cuando tag_length se deriva de él
        (la versión recibida fijaba tag_length=64 directamente, ignorando epsilon_auth).

        NO es una autenticación completa del canal clásico (que
        protegería, mensaje a mensaje, contra un adversario activo tipo
        man-in-the-middle sobre bases/paridades/semillas). Es una
        comprobación de consistencia entre las copias de clave de Alice
        y Bob tras la reconciliación.
        """
        if len(clave_a) == 0 or len(clave_b) == 0:
            return False
        seed = self.entropy.random_seed_int()
        tag_length = self.sec_params.tag_length_efectivo
        tag_a = self.privacy_amplification_toeplitz(clave_a, tag_length, seed)
        tag_b = self.privacy_amplification_toeplitz(clave_b, tag_length, seed)
        return bool(np.array_equal(tag_a, tag_b))


# ============================================================================
# 5. Capa de Seguridad 
# ============================================================================

class SecurityLayer:
    """Capa 3: Evalúa la cota de información y genera el reporte final de seguridad."""

    def __init__(self, sec_params: SecurityParameters):
        self.sec_params = sec_params

    def calculate_lhl_length(
        self,
        n_resto: int,
        e_ph: PhaseErrorEstimate,
        leak_ec: int,
        tag_length: int = 0,
    ) -> int:
        """
        Leftover Hash Lemma (LHL) para Finite-Key:
            l = floor( n_resto * (1 - h2(e_ph)) - leak_ec - tag_length
                       - 2*log2(1/eps_PA) )

        n_resto*(1-h2(e_ph)) estima la entropía mínima de alice_resto
        condicionada a la información física de Eve (supuesto de canal
        simétrico, ver PhaseErrorEstimate). leak_ec y tag_length son la
        regla de cadena de la entropía mínima: CUALQUIER bit adicional
        que se anuncie por el canal público sobre alice_resto reduce esa
        entropía en, como máximo, un bit por cada bit anunciado.

        La versión recibida solo restaba leak_ec. `key_confirmation` calcula su tag
        directamente sobre alice_resto (no sobre una submuestra aparte
        como en Parameter Estimation) y lo hace público al compararlo;
        omitir tag_length aquí sobreestimaba ligeramente longitud_final.
        No es un fallo tan grave como el de Cascade (tag_length son
        unas pocas decenas de bits frente a los miles de longitud de la
        clave), pero la fórmula debe reflejar TODO lo que se anuncia
        sobre alice_resto, no solo la reconciliación.

        Nota: 2*log2(1/eps_PA) es el término de suavizado estándar de la
        versión del Leftover Hash Lemma con información lateral cuántica
        (ver p.ej. Renner 2005; Tomamichel et al. 2011 para la versión
        con entropía mínima suave). No se pretende que esta fórmula sea
        una prueba composable completa (eso exigiría estimar e_ph de
        forma independiente del QBER de bit y propagar epsilon_pe junto
        con epsilon_pa mediante las reglas de cadena correspondientes,
        Fase 3 / punto 22 de la hoja de ruta).
        """
        if n_resto <= 0 or e_ph.value >= UMBRAL_QBER_SEGURIDAD:
            return 0

        h2_eph = float(entropia_binaria(e_ph.value))
        term_entropia = n_resto * (1.0 - h2_eph)
        term_eps = 2.0 * math.log2(1.0 / self.sec_params.epsilon_pa)

        l_val = math.floor(term_entropia - leak_ec - tag_length - term_eps)
        return max(0, l_val)

    def evaluate_and_build(
        self,
        detection: DetectionResult,
        n_tamizada: int,
        pe_data: dict[str, Any] | None,
        bits_revelados_ec: int,
        discrepancias: int,
        auth_ok: bool,
        clave_alice_pa: np.ndarray,
        clave_bob_pa: np.ndarray,
        distancia_km: float | None,  # v5.4: None si el canal no es de fibra
        abort_reason: str | None = None,
    ) -> SecurityReport:
        n_enviados = detection.n_enviados
        click_rate = detection.n_clicks / n_enviados if n_enviados > 0 else 0.0
        dark_rate = detection.n_dark_counts / n_enviados if n_enviados > 0 else 0.0
        double_rate = detection.n_double_clicks / n_enviados if n_enviados > 0 else 0.0

        if abort_reason or pe_data is None:
            # La versión recibida descartaba pe_data incluso cuando SÍ estaba disponible
            # (p.ej. abortos por QBER alto), informando bit_error=0.0 y
            # phase_error_bound=1.0 como placeholders sin relación con lo
            # realmente medido. Si pe_data existe, se usan sus valores
            # reales: quien lea un informe abortado necesita saber CUÁL
            # era el QBER medido, no un valor centinela.
            if pe_data is not None:
                bit_err_abort = pe_data["bit_error"]
                phase_bound_abort = pe_data["phase_error_bound"]
                n_verif_abort = pe_data["n_verificacion"]
            else:
                bit_err_abort = BitErrorEstimate(0.0)
                phase_bound_abort = PhaseErrorEstimate(1.0)
                n_verif_abort = 0

            return SecurityReport(
                distancia_km=distancia_km,
                n_qubits=n_enviados,
                eta_fibra=detection.eta_fibra,
                eta_detector=detection.eta_detector,
                eta_total=detection.eta_total,
                n_clicks=detection.n_clicks,
                n_tamizada=n_tamizada,
                n_verificacion=n_verif_abort,
                bit_error=bit_err_abort,
                phase_error_bound=phase_bound_abort,
                leak_ec_real=0,
                leak_ec_teorico=0.0,
                discrepancias_tras_cascade=discrepancias,
                autenticacion_ok=False,
                abortado=True,
                razon=abort_reason or "Abortado.",
                longitud_clave_final=0,
                tasa_asintotica_bits_por_pulso=0.0,
                tasa_empirica_bits_por_pulso=0.0,
                detector_click_rate=click_rate,
                dark_click_rate=dark_rate,
                double_click_rate=double_rate,
                clave_final_alice=np.array([], dtype=np.uint8),
                clave_final_bob=np.array([], dtype=np.uint8),
            )

        bit_err = pe_data["bit_error"]
        phase_bound = pe_data["phase_error_bound"]
        n_resto = len(pe_data["clave_alice_resto"])

        leak_ec_real = bits_revelados_ec
        leak_ec_teorico = (
            self.sec_params.fec_efficiency
            * n_resto
            * float(entropia_binaria(bit_err.value))
        )

        abort = False
        razon = "Transmisión Segura exitosa."

        if phase_bound.value >= UMBRAL_QBER_SEGURIDAD:
            abort = True
            razon = f"QBER Cota ({phase_bound.value:.4f}) supera umbral ({UMBRAL_QBER_SEGURIDAD})."
        elif not auth_ok:
            abort = True
            razon = "Fallo de autenticación / confirmación de clave."
        elif len(clave_alice_pa) == 0:
            # v5.3: la versión anterior reportaba "Transmisión Segura
            # exitosa" con longitud_clave_final=0 cuando el LHL no dejaba
            # bits (n_resto pequeño, QBER cerca del umbral, leak_ec alto...).
            # Una ejecución que no produce NI UN bit de clave no es un
            # éxito operativo; se marca aborto con causa explícita para que
            # ningún consumidor del informe confunda "seguro" con "útil".
            abort = True
            razon = (
                "LHL no deja bits seguros (longitud final = 0): "
                "n_resto insuficiente para el leak_ec + tag + término de "
                "suavizado con este QBER."
            )

        final_len = len(clave_alice_pa) if not abort else 0
        tasa_empirica = final_len / n_enviados if n_enviados > 0 else 0.0

        # Tasa asintótica teórica
        h_q = float(entropia_binaria(bit_err.value))
        tasa_asintotica = (n_tamizada / n_enviados) * max(
            0.0, 1.0 - (1.0 + self.sec_params.fec_efficiency) * h_q
        )

        return SecurityReport(
            distancia_km=distancia_km,
            n_qubits=n_enviados,
            eta_fibra=detection.eta_fibra,
            eta_detector=detection.eta_detector,
            eta_total=detection.eta_total,
            n_clicks=detection.n_clicks,
            n_tamizada=n_tamizada,
            n_verificacion=pe_data["n_verificacion"],
            bit_error=bit_err,
            phase_error_bound=phase_bound,
            leak_ec_real=leak_ec_real,
            leak_ec_teorico=leak_ec_teorico,
            discrepancias_tras_cascade=discrepancias,
            autenticacion_ok=auth_ok,
            abortado=abort,
            razon=razon,
            longitud_clave_final=final_len,
            tasa_asintotica_bits_por_pulso=tasa_asintotica,
            tasa_empirica_bits_por_pulso=tasa_empirica,
            detector_click_rate=click_rate,
            dark_click_rate=dark_rate,
            double_click_rate=double_rate,
            clave_final_alice=clave_alice_pa if not abort else np.array([], dtype=np.uint8),
            clave_final_bob=clave_bob_pa if not abort else np.array([], dtype=np.uint8),
        )


# ============================================================================
# 6. Simulador Principal (Facade de Orquestación)
# ============================================================================

class BB84Simulator:
    """Orquestador principal desacoplado que ejecuta la simulación de 3 capas."""

    def __init__(
        self,
        entropy: EntropySource,
        sec_params: SecurityParameters | None = None,
    ):
        self.entropy = entropy
        self.sec_params = sec_params or SecurityParameters()
        self.classical_layer = ClassicalLayer(self.entropy, self.sec_params)
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
        if n_qubits <= 0:
            raise ValueError("n_qubits debe ser positivo.")

        valores_fibra = {
            "distancia_km": distancia_km,
            "atenuacion_db_km": atenuacion_db_km,
            "eta_detector": eta_detector,
            "prob_dark_count": prob_dark_count,
            "qber_intrinseco": qber_intrinseco,
            "eve": eve,
            "eve_fraction": eve_fraction,
        }

        if channel is not None:
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
            channel = FiberChannel(entropy=self.entropy, **valores_fibra)

        # None si `channel` no es (ni envuelve) un canal de fibra.
        distancia_km_reporte = getattr(channel, "distancia_km", None)

        # 1. Configuración de la Capa Cuántica
        alice = Alice(self.entropy)
        bob = Bob(self.entropy)
        quantum_layer = QuantumLayer(alice, bob, channel)

        # Transmisión cuántica
        detection = quantum_layer.execute(n_qubits)

        # 2. Métodos Privados de la Fase Clásica
        clave_alice, clave_bob = self.classical_layer.sifting(detection)
        n_tamizada = len(clave_alice)

        if n_tamizada < n_min_tamizados:
            return self.security_layer.evaluate_and_build(
                detection, n_tamizada, None, 0, 0, False,
                np.array([]), np.array([]), distancia_km_reporte,
                abort_reason="Insuficientes bits tamizados."
            )

        # Parameter Estimation
        pe_data = self.classical_layer.parameter_estimation(
            clave_alice, clave_bob, fraccion_verificacion
        )

        # Salida temprana: si el QBER YA supera el umbral de seguridad no
        # tiene sentido gastar Cascade + amplificación de privacidad en
        # una ejecución que se va a abortar de todas formas (la versión recibida los
        # ejecutaba siempre y descartaba el resultado al final).
        if pe_data["phase_error_bound"].value >= UMBRAL_QBER_SEGURIDAD:
            return self.security_layer.evaluate_and_build(
                detection, n_tamizada, pe_data, 0, 0, False,
                np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), distancia_km_reporte,
                abort_reason=(
                    f"QBER Cota ({pe_data['phase_error_bound'].value:.4f}) "
                    f"supera umbral ({UMBRAL_QBER_SEGURIDAD}) -> posible espía."
                ),
            )

        alice_resto = pe_data["clave_alice_resto"]
        bob_resto = pe_data["clave_bob_resto"]

        # Error Correction
        bob_reconciliado, leak_ec, discrepancias = (
            self.classical_layer.error_correction_cascade(
                alice_resto,
                bob_resto,
                qber_estimado=pe_data["bit_error"].value,
                n_pasadas=n_pasadas_cascade,
            )
        )

        # Key Confirmation
        auth_ok = self.classical_layer.key_confirmation(alice_resto, bob_reconciliado)

        # Privacy Amplification (Leftover Hash Lemma). Se descuenta también
        # tag_length_efectivo: key_confirmation revela ese hash sobre
        # alice_resto por el canal público (ver docstring de
        # calculate_lhl_length).
        target_len = self.security_layer.calculate_lhl_length(
            n_resto=len(alice_resto),
            e_ph=pe_data["phase_error_bound"],
            leak_ec=leak_ec,
            tag_length=self.sec_params.tag_length_efectivo,
        )

        seed_pa = self.entropy.random_seed_int()
        clave_alice_pa = self.classical_layer.privacy_amplification_toeplitz(
            alice_resto, target_len, seed_pa
        )
        clave_bob_pa = self.classical_layer.privacy_amplification_toeplitz(
            bob_reconciliado, target_len, seed_pa
        )

        # 3. Capa de Seguridad (Construcción del informe)
        return self.security_layer.evaluate_and_build(
            detection=detection,
            n_tamizada=n_tamizada,
            pe_data=pe_data,
            bits_revelados_ec=leak_ec,
            discrepancias=discrepancias,
            auth_ok=auth_ok,
            clave_alice_pa=clave_alice_pa,
            clave_bob_pa=clave_bob_pa,
            distancia_km=distancia_km_reporte,
        )


# ============================================================================
# 7. Suite de Tests Estadísticos (Aseguramiento de Calidad)
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
    entropy = EntropySource.simulation(seed=42)
    sec_params = SecurityParameters()
    sim = BB84Simulator(entropy, sec_params)

    # Test 1: Sifting (~50%)
    res_ideal = sim.run(n_qubits=100_000, qber_intrinseco=0.0, eta_detector=1.0)
    ratio_sift = res_ideal.n_tamizada / 100_000
    assert 0.48 < ratio_sift < 0.52, f"Falló Sifting Ratio: {ratio_sift}"
    print(f" [PASS] Sifting Rate Ratio ideal: {ratio_sift:.4f} (~0.50)")

    # Test 2: QBER con Intercept-Resend completo (debe dar ~25%)
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
    res_canal = sim.run(
        n_qubits=100_000,
        distancia_km=10.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=0.0,
    )
    eta_teorica = 10 ** (-0.2 * 10 / 10)  # 0.6309
    eta_medida = res_canal.detector_click_rate
    assert math.isclose(eta_medida, eta_teorica, rel_tol=0.05)
    print(f" [PASS] Canal Attenuation Click Rate: {eta_medida:.4f} (Teórico: {eta_teorica:.4f})")

    # Test 4: Uniformidad de bits y bases de Alice (antes de cualquier
    # efecto del canal). Media Y chi-cuadrado de bondad de ajuste: la
    # media sola no detectaría, p.ej., una anti-correlación bit<->base
    # que dejara ambas medias en ~0.5 pero la fuente sesgada.
    alice_test = Alice(EntropySource.simulation(seed=123))
    n_unif = 200_000
    paquete = alice_test.prepare(n_unif)
    media_bits = float(np.mean(paquete.bits))
    media_bases = float(np.mean(paquete.bases))
    assert 0.49 < media_bits < 0.51, f"Falló uniformidad de bits: {media_bits}"
    assert 0.49 < media_bases < 0.51, f"Falló uniformidad de bases: {media_bases}"
    print(f" [PASS] Uniformidad de bits de Alice: {media_bits:.4f} (~0.50)")
    print(f" [PASS] Uniformidad de bases de Alice: {media_bases:.4f} (~0.50)")

    conteo_bits = np.bincount(paquete.bits, minlength=2)
    chi2_bits = float(np.sum((conteo_bits - n_unif / 2) ** 2) / (n_unif / 2))
    # 1 grado de libertad; chi2 > 10.83 rechazaría uniformidad a p<0.001
    assert chi2_bits < 10.83, f"Chi-cuadrado de bits sospechosamente alto: {chi2_bits:.3f}"
    print(f" [PASS] Chi-cuadrado uniformidad de bits: {chi2_bits:.3f} (< 10.83 a p<0.001)")

    # v5.4: el chi-cuadrado de bits por sí solo (arriba) no cubría lo que
    # el comentario de este test llevaba tiempo prometiendo -- detectar
    # una anti-correlación bit<->base con ambas marginales en ~0.5.
    # Se añade el chi-cuadrado de uniformidad de BASES (que faltaba) y,
    # sobre todo, un chi-cuadrado de independencia sobre la tabla de
    # contingencia 2x2 bit x base: es la única de las tres pruebas que
    # de verdad detectaría esa anti-correlación.
    conteo_bases = np.bincount(paquete.bases, minlength=2)
    chi2_bases = float(np.sum((conteo_bases - n_unif / 2) ** 2) / (n_unif / 2))
    assert chi2_bases < 10.83, f"Chi-cuadrado de bases sospechosamente alto: {chi2_bases:.3f}"
    print(f" [PASS] Chi-cuadrado uniformidad de bases: {chi2_bases:.3f} (< 10.83 a p<0.001)")

    tabla = np.zeros((2, 2), dtype=int)
    np.add.at(tabla, (paquete.bits, paquete.bases), 1)
    esperado = (
        tabla.sum(axis=1, keepdims=True)
        * tabla.sum(axis=0, keepdims=True)
        / tabla.sum()
    )
    chi2_independencia = float(np.sum((tabla - esperado) ** 2 / esperado))
    # También 1 grado de libertad: una tabla 2x2 tiene (2-1)*(2-1) = 1.
    assert chi2_independencia < 10.83, (
        f"Bits y bases no parecen independientes: chi²={chi2_independencia:.3f}"
    )
    print(f" [PASS] Chi-cuadrado independencia bit-base: {chi2_independencia:.3f} (< 10.83 a p<0.001)")

    # Test 5: Tasa de dark counts. A 500 km / 0.2 dB/km, eta_fibra ~ 1e-10:
    # prácticamente ningún fotón real sobrevive, así que casi todos los
    # clicks observados deben venir de dark counts (click_rate ~ prob_dark_count).
    res_dark = sim.run(
        n_qubits=200_000,
        distancia_km=500.0,
        atenuacion_db_km=0.2,
        eta_detector=1.0,
        prob_dark_count=1e-3,
        qber_intrinseco=0.0,
    )
    assert math.isclose(res_dark.detector_click_rate, 1e-3, rel_tol=0.25)
    print(f" [PASS] Dark count rate a canal saturado: {res_dark.detector_click_rate:.5f} (objetivo ~0.00100)")

    # Test 6 (regresión): Cascade debe corregir el 100% de los errores
    # inyectados, para un barrido de QBER realista y varias semillas.
    # Ver el docstring de esta función para el motivo.
    rng_test = np.random.default_rng(2026)
    n_test = 20_000
    for qber_test in [0.01, 0.03, 0.05, 0.08]:
        alice_k = rng_test.integers(0, 2, size=n_test)
        bob_k = alice_k.copy()
        flips = rng_test.random(n_test) < qber_test
        bob_k[flips] ^= 1
        capa_test = ClassicalLayer(
            EntropySource.simulation(int(rng_test.integers(0, 2**31))), sec_params
        )
        _, _, discrepancias = capa_test.error_correction_cascade(
            alice_k, bob_k, qber_estimado=qber_test, n_pasadas=4
        )
        assert discrepancias == 0, (
            f"Cascade dejó {discrepancias} discrepancias sin corregir a "
            f"QBER={qber_test:.1%} (n={n_test})"
        )
    print(" [PASS] Cascade corrige el 100% de los errores inyectados (QBER 1%-8%, n=20 000)")

    print("=== Todos los tests pasaron correctamente. ===\n")
