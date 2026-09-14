"""
Simulador BB84 (Distribución de Claves Cuánticas).

Módulo principal para la simulación del protocolo QKD BB84, incluyendo
modelado de canales cuánticos con ruido, detección de intromisión (Eve),
reconciliación de información por Cascade y amplificación de privacidad.

"""

from __future__ import annotations

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


def _validar_prob01(nombre: str, valor: Any) -> None:
    """Valida que `valor` sea un real en [0, 1] (probabilidad o
    eficiencia de detector cerrada, incluyendo los extremos). Se
    reutiliza en SecurityParameters y en todos los ChannelModel
    concretos para no duplicar el mismo chequeo en cada clase.

    No hace falta comprobar isfinite() aparte: al estar acotado por
    AMBOS lados con una comparación encadenada, NaN e inf ya quedan
    fuera por cortocircuito (`0.0 <= nan` y `inf <= 1.0` son ambas
    False), a diferencia de _validar_no_negativo más abajo."""
    if isinstance(valor, bool) or not isinstance(valor, Real) or not 0.0 <= valor <= 1.0:
        raise ValueError(f"{nombre} debe estar en el intervalo [0, 1].")


def _validar_epsilon(nombre: str, valor: Any) -> None:
    """Valida que `valor` sea un real en (0, 1), abierto en ambos
    extremos: un epsilon de seguridad de exactamente 0 (confianza
    absoluta) o 1 (ninguna confianza) no tiene sentido en el marco
    composable de estos parámetros. Igual que _validar_prob01, al estar
    acotado por ambos lados no necesita un isfinite() aparte."""
    if isinstance(valor, bool) or not isinstance(valor, Real) or not 0.0 < valor < 1.0:
        raise ValueError(f"{nombre} debe estar en el intervalo (0, 1).")


def _validar_no_negativo(nombre: str, valor: Any) -> None:
    """Valida que `valor` sea un real FINITO >= 0 (distancias en km,
    atenuaciones en dB/km o dB -- nunca negativas).

    v5.4.1: a diferencia de _validar_prob01/_validar_epsilon, este
    chequeo solo está acotado por UN lado (`valor < 0.0`), y NaN no
    cumple ninguna comparación ordinaria -- ni siquiera `nan < 0.0`,
    que da False --, así que un distancia_km=nan se colaba como
    "válido". +inf tampoco cumple `< 0.0` y se colaba igual, pese a no
    tener sentido físico como distancia o atenuación. Se exige
    isfinite() explícitamente en vez de fiarse solo de la cota
    inferior."""
    if (
        isinstance(valor, bool)
        or not isinstance(valor, Real)
        or not math.isfinite(valor)
        or valor < 0.0
    ):
        raise ValueError(f"{nombre} debe ser un real finito >= 0.")


@dataclass(frozen=True)
class SecurityParameters:
    """
    Parámetros de seguridad unificados según el marco composable moderno.

    En la versión recibida, `epsilon_auth` y `epsilon_ec` se declaraban pero no se
    leían en ningún cálculo (`tag_length` se usaba directamente, sin
    relación con `epsilon_auth`). Ahora `tag_length` es opcional: si no
    se fija explícitamente, se DERIVA de `epsilon_auth` mediante
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
        infla la longitud de clave declarada segura, no un simple
        capricho de "entrada rara". Los cuatro epsilons son términos de
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

    LIMITACIÓN DOCUMENTADA (ya presente en la versión recibida, aquí explicitada): aquí
    e_ph se toma igual a la cota de Serfling del error de BIT
    (supuesto de canal simétrico/depolarizante). Para un adversario
    general e_ph podría estimarse de forma independiente; se mantiene
    esta clase separada de BitErrorEstimate precisamente para que ese
    cambio, el día de mañana, no obligue a tocar el resto del pipeline.
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
# 1. Fuente de Entropía
# ============================================================================

class EntropySource:
    """
    Capa de abstracción para la aleatoriedad (PRNG / CSPRNG).

    Nota de diseño: no TODA la aleatoriedad del protocolo necesita ser
    criptográfica. La elección de bit/base de Alice y Bob sí (termina,
    parcialmente, en la clave secreta). Las semillas PÚBLICAS de Cascade
    y de la matriz de Toeplitz de amplificación de privacidad, en
    cambio, no necesitan ser secretas por el Leftover Hash Lemma: solo
    necesitan elegirse con independencia de la información de Eve. Por
    eso la expansión semilla->estructura se hace siempre con
    `numpy.random.default_rng(semilla)` (determinista, público,
    reproducible por ambas partes), mientras que la semilla en sí puede
    generarse con este EntropySource en cualquiera de los dos modos.
    Ver `ClassicalLayer.error_correction_cascade` y
    `privacy_amplification_toeplitz`.
    """

    def __init__(self, mode: str = "simulation", seed: int | None = None):
        mode = mode.lower()
        if mode not in {"simulation", "crypto"}:
            raise ValueError("mode debe ser 'simulation' o 'crypto'.")

        self.mode = mode
        self.seed = seed

        if mode == "simulation":
            self._rng = np.random.default_rng(seed)
            self._crypto = None
        else:
            self._rng = None
            self._crypto = secrets.SystemRandom()

    @classmethod
    def simulation(cls, seed: int | None = None) -> EntropySource:
        return cls("simulation", seed)

    @classmethod
    def crypto(cls) -> EntropySource:
        return cls("crypto")

    @staticmethod
    def _bits_criptograficos(n: int) -> np.ndarray:
        """n bits uniformes de os.urandom, vectorizado con unpackbits.
        Mismo CSPRNG del SO que secrets.SystemRandom, pero sin el coste
        de un bucle Python por bit."""
        n_bytes = (n + 7) // 8
        crudo = np.frombuffer(os.urandom(n_bytes), dtype=np.uint8)
        return np.unpackbits(crudo)[:n].astype(np.int64)

    @staticmethod
    def _uniforme01_criptografico(n: int) -> np.ndarray:
        """n floats U[0,1) a partir de enteros de 32 bits de os.urandom."""
        crudo = np.frombuffer(os.urandom(n * 4), dtype=np.uint32)
        return crudo.astype(np.float64) / np.float64(2**32)

    def integers(self, low: int, high: int | None = None, size=None):
        if self.mode == "simulation":
            return self._rng.integers(low, high, size=size)

        if high is None:
            high = low
            low = 0

        if size is None:
            return self._crypto.randrange(low, high)

        n = int(np.prod(size)) if isinstance(size, tuple) else int(size)

        if low == 0 and high == 2:
            # Camino rápido, vectorizado: es el caso dominante (bits de
            # Alice/Bob, y de hecho el único que este simulador ejercita
            # a escala de miles/millones de elementos).
            values = self._bits_criptograficos(n)
        else:
            values = np.array([self._crypto.randrange(low, high) for _ in range(n)])

        return values.reshape(size) if isinstance(size, tuple) else values.astype(np.int64)

    def random(self, size=None):
        if self.mode == "simulation":
            return self._rng.random(size=size)

        if size is None:
            return self._crypto.random()

        n = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        values = self._uniforme01_criptografico(n)
        return values.reshape(size) if isinstance(size, tuple) else values

    def permutation(self, n: int) -> np.ndarray:
        if self.mode == "simulation":
            return self._rng.permutation(n)

        values = list(range(n))
        self._crypto.shuffle(values)
        return np.asarray(values, dtype=np.int64)

    def choice(self, a, size=None, replace=True):
        """
        v5.4: la rama crypto no soportaba `size` como tupla -- fallaba
        con TypeError en la comparación `size > len(values)` en cuanto
        `size` no era un entero (p.ej. size=(1, 1), o cualquier forma
        multidimensional). También se generaliza `a` para aceptar un
        entero (elegir de range(a)), igual que np.random.Generator.choice.

        El muestreo CON reemplazo sigue siendo una llamada a
        self._crypto.choice() por elemento (que resuelve internamente
        con _randbelow(), es decir con rechazo, no con el patrón
        floor(random()*n) de random.choices()): es la opción más
        cuidadosa para muestreo uniforme criptográfico, aunque
        random.choices() habría evitado el bucle Python.
        """
        if self.mode == "simulation":
            return self._rng.choice(a, size=size, replace=replace)

        values = list(range(a)) if isinstance(a, (int, np.integer)) else list(a)

        if size is None:
            return self._crypto.choice(values)

        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        if not all(isinstance(d, (int, np.integer)) and d >= 0 for d in shape):
            raise ValueError("size debe contener enteros no negativos.")

        n = int(np.prod(shape)) if shape else 1

        if not replace and n > len(values):
            raise ValueError("No se pueden elegir tantos elementos sin reemplazo.")

        if replace:
            selected = [self._crypto.choice(values) for _ in range(n)]
        else:
            selected = self._crypto.sample(values, n)

        return np.asarray(selected).reshape(shape)

    def random_seed_int(self, upper: int = 2**63 - 1) -> int:
        if self.mode == "simulation":
            return int(self._rng.integers(0, upper))
        return self._crypto.randrange(0, upper)

    def poisson(self, lam) -> np.ndarray:
        """
        Muestrea Poisson(lam), donde `lam` puede ser un array (una
        intensidad distinta por pulso, como en una fuente WCP con
        varias intensidades señuelo). Necesario para
        decoy_wcp.AliceWCP.

        Modo simulation: delega en numpy (algoritmo PTRS, eficiente
        para cualquier lam).

        Modo crypto: algoritmo de Knuth (cuenta cuántos uniformes caben
        antes de que su producto caiga por debajo de e^-lam),
        vectorizado por elemento con os.urandom. Coste esperado
        O(lam+1) por elemento -- perfectamente adecuado para las
        intensidades de decoy-state (mu <= ~1 típicamente), no
        pensado para lam grande.
        """
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=np.float64))
        escalar = np.ndim(lam) == 0

        if self.mode == "simulation":
            resultado = self._rng.poisson(lam_arr)
        else:
            n = lam_arr.shape[0]
            L = np.exp(-lam_arr)
            k = np.zeros(n, dtype=np.int64)
            p = np.ones(n, dtype=np.float64)
            activos = lam_arr > 1e-15  # Poisson(0) = delta en 0, no hace falta muestrear
            while np.any(activos):
                idx = np.flatnonzero(activos)
                k[idx] += 1
                p[idx] *= self._uniforme01_criptografico(len(idx))
                activos[idx] = p[idx] > L[idx]
            resultado = k - 1

        return int(resultado[0]) if escalar else resultado.astype(np.int64)


# ============================================================================
# 2. Interfaces de Entidades y Ataques (Extensibilidad)
# ============================================================================

@dataclass(frozen=True)
class QuantumPacket:
    """
    Nota: `frozen=True` impide REASIGNAR bits/bases tras construir el
    paquete (`packet.bits = otro_array` lanzaría FrozenInstanceError),
    pero no hace los arrays de NumPy internamente inmutables
    (`packet.bits[0] = 1` seguiría funcionando). Para inmutabilidad
    profunda habría que exponer copias de solo lectura
    (`array.setflags(write=False)`); no se hace aquí para no penalizar
    el rendimiento en el camino caliente de miles de qubits.
    """
    bits: np.ndarray
    bases: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    bits_alice: np.ndarray
    bases_alice: np.ndarray
    bits_bob: np.ndarray
    bases_bob: np.ndarray
    eta_fibra: float
    eta_detector: float
    eta_total: float
    n_enviados: int
    n_clicks: int
    n_fotones_detectados: int
    n_dark_counts: int
    n_double_clicks: int


class EveStrategy(ABC):
    """Interfaz abstracta para estrategias de ataque de Eve."""

    @abstractmethod
    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float
    ) -> QuantumPacket:
        pass


class InterceptResendEve(EveStrategy):
    """Ataque activo Intercept-Resend."""

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction debe estar entre 0 y 1.")

        n = len(packet.bits)
        intercept = entropy.random(n) < fraction

        eve_bases = entropy.integers(0, 2, size=n)
        eve_random = entropy.integers(0, 2, size=n)

        eve_results = np.where(packet.bases == eve_bases, packet.bits, eve_random)
        bits_out = np.where(intercept, eve_results, packet.bits)
        bases_out = np.where(intercept, eve_bases, packet.bases)

        return QuantumPacket(bits=bits_out, bases=bases_out)


class PassiveEve(EveStrategy):
    """Eve no interactúa con los qubits en vuelo."""

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        return packet


class RandomEve(EveStrategy):
    """
    Decorador de activación probabilística sobre otra EveStrategy.

    Distinto de InterceptResendEve(fraction<1): `fraction` allí decide,
    POR CADA FOTÓN, si Eve lo intercepta (Eve está "siempre presente"
    pero actúa parcialmente). RandomEve decide, POR EJECUCIÓN, si Eve
    está presente en absoluto -- por ejemplo para explorar un modelo en
    el que el adversario solo ataca una fracción de las sesiones QKD
    (útil para estudios de Monte Carlo sobre la probabilidad de detectar
    un espía intermitente). Se implementa como decorador para no
    duplicar lógica de ataque: envuelve cualquier EveStrategy existente.
    """

    def __init__(self, estrategia_base: EveStrategy, probabilidad_activacion: float = 0.5):
        if not 0.0 <= probabilidad_activacion <= 1.0:
            raise ValueError("probabilidad_activacion debe estar entre 0 y 1.")
        self.estrategia_base = estrategia_base
        self.probabilidad_activacion = probabilidad_activacion

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        activa = entropy.random(None) < self.probabilidad_activacion
        if activa:
            return self.estrategia_base.attack(packet, entropy, fraction)
        return packet


class CollectiveAttack(EveStrategy):
    """
    NO IMPLEMENTADA a nivel de simulación de bits -- placeholder de
    interfaz intencionado, no un ataque real.

    Un ataque colectivo/coherente general implica que Eve aplica una
    operación unitaria conjunta sobre su propio sistema cuántico y cada
    qubit en tránsito, y GUARDA su sistema en memoria cuántica para
    medirlo más tarde (incluso después de conocer las bases públicas),
    de forma óptima sobre el estado conjunto. Esto no tiene una
    descripción clásica bit-a-bit como intercept-resend: su efecto se
    acota mediante desigualdades de teoría de la información (p. ej.
    Devetak-Winter 2005, o el marco de entropía mínima suave de Renner
    2005), no simulando qué "mide" Eve pulso a pulso con un rng.integers.

    Instanciar esta clase y llamar a attack() lanza NotImplementedError
    explícito, en vez de fingir un comportamiento físicamente incorrecto
    (p. ej. simular "Eve mide en una base aleatoria" y llamarlo
    "colectivo", que sería sencillamente InterceptResendEve con otro
    nombre). El lugar correcto para modelar el efecto de un ataque
    colectivo en ESTE simulador es endureciendo la fórmula de longitud
    final de clave (SecurityLayer.calculate_lhl_length) con una cota
    válida contra ataques generales -- no esta clase.
    """

    def attack(
        self, packet: QuantumPacket, entropy: EntropySource, fraction: float = 1.0
    ) -> QuantumPacket:
        raise NotImplementedError(
            "CollectiveAttack no es simulable a nivel de bits/bases. "
            "Ver el docstring de la clase para el enfoque correcto "
            "(endurecer la cota de seguridad, no simular la medida de Eve)."
        )


class ChannelModel(ABC):
    """Interfaz abstracta para el canal físico cuántico."""

    @abstractmethod
    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        pass


# ============================================================================
# 3. Capa Cuántica (Quantum Layer)
# ============================================================================

class Alice:
    """Emisor BB84."""

    def __init__(self, entropy: EntropySource):
        self.entropy = entropy

    def prepare(self, n_qubits: int) -> QuantumPacket:
        bits = self.entropy.integers(0, 2, size=n_qubits)
        bases = self.entropy.integers(0, 2, size=n_qubits)
        return QuantumPacket(bits=bits, bases=bases)


class Bob:
    """Receptor BB84."""

    def __init__(self, entropy: EntropySource):
        self.entropy = entropy

    def choose_bases(self, n_qubits: int) -> np.ndarray:
        return self.entropy.integers(0, 2, size=n_qubits)

    def measure(self, packet: QuantumPacket, bases_bob: np.ndarray) -> np.ndarray:
        coinciden = packet.bases == bases_bob
        aleatorios = self.entropy.integers(0, 2, size=len(packet.bits))
        return np.where(coinciden, packet.bits, aleatorios)


def _detectar_en_bob(
    packet: QuantumPacket,
    packet_canal: QuantumPacket,
    bob: Bob,
    entropy: EntropySource,
    eta_fibra: float,
    eta_detector: float,
    prob_dark_count: float,
    qber_intrinseco: float = 0.0,
    prob_depolarizacion: float = 0.0,
) -> DetectionResult:
    """
    Física de detección COMPARTIDA por todos los ChannelModel: pérdida
    (eta_fibra*eta_detector), dark counts, doble-click ambiguo, medida
    de Bob en su base, y dos fuentes de error opcionales:
      - qber_intrinseco: flip probabilístico del resultado de Bob,
        aplicado a TODA detección real, coincida o no la base (v5.3:
        el docstring anterior decía "solo cuando la base coincidía",
        pero el código nunca condicionó por base -- da igual para el
        QBER tamizado, porque los casos de base distinta se descartan
        en el sifting y su resultado ya era uniforme, pero el texto
        debe describir lo que el código hace).
      - prob_depolarizacion: sustitución por un bit aleatorio uniforme
        independientemente de la base (canal despolarizante,
        rho -> (1-p)rho + p*I/2).
    Antes de esta revisión, FiberChannel implementaba esta lógica en su
    propio transmit(); al añadir DepolarizingChannel/FreeSpaceChannel se
    hubiera triplicado. Cada ChannelModel concreto solo calcula SU
    eta_fibra (con el modelo físico que le corresponda) y llama aquí.
    """
    eta_total = eta_fibra * eta_detector
    n = len(packet.bits)

    llega_foton = entropy.random(n) < eta_total
    hay_dark_count = entropy.random(n) < prob_dark_count
    hay_click = llega_foton | hay_dark_count
    doble_click = llega_foton & hay_dark_count

    bases_bob = bob.choose_bases(n)
    resultado_real = bob.measure(packet_canal, bases_bob)

    if qber_intrinseco > 0:
        flips = entropy.random(n) < qber_intrinseco
        resultado_real = np.where(flips, 1 - resultado_real, resultado_real)

    if prob_depolarizacion > 0:
        despolariza = entropy.random(n) < prob_depolarizacion
        aleatorio = entropy.integers(0, 2, size=n)
        resultado_real = np.where(despolariza, aleatorio, resultado_real)

    resultado_dark = entropy.integers(0, 2, size=n)
    bits_bob = np.where(
        doble_click,
        entropy.integers(0, 2, size=n),
        np.where(llega_foton, resultado_real, resultado_dark),
    )

    idx_click = np.flatnonzero(hay_click)
    return DetectionResult(
        bits_alice=packet.bits[idx_click],
        bases_alice=packet.bases[idx_click],
        bits_bob=bits_bob[idx_click],
        bases_bob=bases_bob[idx_click],
        eta_fibra=eta_fibra,
        eta_detector=eta_detector,
        eta_total=eta_total,
        n_enviados=n,
        n_clicks=len(idx_click),
        n_fotones_detectados=int(np.sum(llega_foton)),
        n_dark_counts=int(np.sum(hay_dark_count)),
        n_double_clicks=int(np.sum(doble_click)),
    )


class FiberChannel(ChannelModel):
    """Modelo de canal de fibra óptica con atenuación y ruido."""

    def __init__(
        self,
        entropy: EntropySource,
        distancia_km: float = 0.0,
        atenuacion_db_km: float = 0.2,
        eta_detector: float = 0.2,
        prob_dark_count: float = 1e-6,
        qber_intrinseco: float = 0.01,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        # v5.4: distancia_km negativa producía eta_fibra = 10**(+algo) > 1
        # (una "ganancia" físicamente absurda); ninguno de los demás
        # parámetros se validaba tampoco.
        _validar_no_negativo("distancia_km", distancia_km)
        _validar_no_negativo("atenuacion_db_km", atenuacion_db_km)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("qber_intrinseco", qber_intrinseco)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.distancia_km = distancia_km
        self.atenuacion_db_km = atenuacion_db_km
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.qber_intrinseco = qber_intrinseco
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def eta_fibra(self) -> float:
        return 10 ** (-self.atenuacion_db_km * self.distancia_km / 10.0)

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet, packet_canal, bob, self.entropy,
            eta_fibra=self.eta_fibra, eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            qber_intrinseco=self.qber_intrinseco,
        )


class DepolarizingChannel(ChannelModel):
    """
    Canal simplificado SIN modelo de pérdida por distancia: toda la
    degradación se modela como ruido despolarizante puro, con
    probabilidad fija `prob_depolarizacion` de sustituir el resultado
    de Bob por un bit aleatorio uniforme (independiente de la base).

    Útil para: (a) enlaces ya compensados en potencia, donde el error
    dominante no es la pérdida de fotones sino el ruido de polarización
    (birrefringencia residual, imperfecciones de alineación); (b)
    aislar en pruebas el efecto del ruido de bit del de la pérdida, que
    en FiberChannel están mezclados en la misma ejecución.
    """

    def __init__(
        self,
        entropy: EntropySource,
        prob_depolarizacion: float = 0.02,
        eta_detector: float = 1.0,
        prob_dark_count: float = 0.0,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        _validar_prob01("prob_depolarizacion", prob_depolarizacion)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.prob_depolarizacion = prob_depolarizacion
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def eta_fibra(self) -> float:
        return 1.0  # sin modelo de pérdida por distancia

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet, packet_canal, bob, self.entropy,
            eta_fibra=self.eta_fibra, eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            prob_depolarizacion=self.prob_depolarizacion,
        )


class FreeSpaceChannel(ChannelModel):
    """
    Canal simplificado para un enlace de espacio libre (línea de visión
    directa), parametrizado por el ángulo de elevación en vez de una
    distancia de fibra.

    MODELO DELIBERADAMENTE SIMPLIFICADO -- no pretende precisión de
    ingeniería de un enlace real: transmitancia atmosférica tipo "ley
    de la secante" (masa de aire ~ 1/sin(elevación), como en Beer-
    Lambert) más una pérdida geométrica fija de apuntamiento/divergencia
    del haz. Un modelo realista necesitaría, como mínimo: parámetro de
    Fried r0 y escintilación por turbulencia, pérdida de apuntamiento
    dependiente del tiempo, ruido de fondo diurno/nocturno según el
    campo de visión del receptor, y la geometría orbital si el enlace es
    con un satélite en movimiento (ver Fase 3 de la hoja de ruta). Esta
    clase existe para demostrar que ChannelModel admite un canal
    cualitativamente distinto de la fibra sin tocar QuantumLayer ni
    ClassicalLayer -- no como sustituto de un simulador de enlace
    óptico-espacial real.
    """

    def __init__(
        self,
        entropy: EntropySource,
        elevacion_grados: float = 45.0,
        atenuacion_cenital_db: float = 3.0,
        perdida_apuntamiento_db: float = 2.0,
        eta_detector: float = 0.5,
        prob_dark_count: float = 1e-5,
        qber_intrinseco: float = 0.01,
        eve: EveStrategy | None = None,
        eve_fraction: float = 1.0,
    ):
        if not 0.0 < elevacion_grados <= 90.0:
            raise ValueError("elevacion_grados debe estar en (0, 90].")
        _validar_no_negativo("atenuacion_cenital_db", atenuacion_cenital_db)
        _validar_no_negativo("perdida_apuntamiento_db", perdida_apuntamiento_db)
        _validar_prob01("eta_detector", eta_detector)
        _validar_prob01("prob_dark_count", prob_dark_count)
        _validar_prob01("qber_intrinseco", qber_intrinseco)
        _validar_prob01("eve_fraction", eve_fraction)
        self.entropy = entropy
        self.elevacion_grados = elevacion_grados
        self.atenuacion_cenital_db = atenuacion_cenital_db
        self.perdida_apuntamiento_db = perdida_apuntamiento_db
        self.eta_detector = eta_detector
        self.prob_dark_count = prob_dark_count
        self.qber_intrinseco = qber_intrinseco
        self.eve = eve
        self.eve_fraction = eve_fraction

    @property
    def masa_de_aire(self) -> float:
        """Aproximación de secante plana; se degrada cerca del horizonte
        (elevación -> 0), aceptable para elevación >= 10-15 grados."""
        return 1.0 / math.sin(math.radians(self.elevacion_grados))

    @property
    def eta_fibra(self) -> float:
        # Nombre heredado de la interfaz común (así SecurityLayer /
        # SecurityReport no necesitan distinguir el tipo de canal); aquí
        # representa la transmitancia atmosférica total.
        atenuacion_db = (
            self.atenuacion_cenital_db * self.masa_de_aire
            + self.perdida_apuntamiento_db
        )
        return 10 ** (-atenuacion_db / 10.0)

    @property
    def eta_total(self) -> float:
        return self.eta_fibra * self.eta_detector

    def transmit(self, packet: QuantumPacket, bob: Bob) -> DetectionResult:
        packet_canal = packet
        if self.eve is not None:
            packet_canal = self.eve.attack(
                packet, self.entropy, fraction=self.eve_fraction
            )
        return _detectar_en_bob(
            packet, packet_canal, bob, self.entropy,
            eta_fibra=self.eta_fibra, eta_detector=self.eta_detector,
            prob_dark_count=self.prob_dark_count,
            qber_intrinseco=self.qber_intrinseco,
        )


class QuantumLayer:
    """Capa 1: Responsable de la ejecución del hardware físico cuántico."""

    def __init__(self, alice: Alice, bob: Bob, channel: ChannelModel):
        self.alice = alice
        self.bob = bob
        self.channel = channel

    def execute(self, n_qubits: int) -> DetectionResult:
        packet = self.alice.prepare(n_qubits)
        return self.channel.transmit(packet, self.bob)


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
# 5. Capa de Seguridad y Reporte (Security Layer)
# ============================================================================

@dataclass(frozen=True)
class SecurityReport:
    """Punto 6 de la hoja de ruta: resultado inmutable una vez construido."""
    distancia_km: float | None  # v5.4: None si el canal no tiene esa noción
    n_qubits: int
    eta_fibra: float
    eta_detector: float
    eta_total: float
    n_clicks: int
    n_tamizada: int
    n_verificacion: int
    bit_error: BitErrorEstimate
    phase_error_bound: PhaseErrorEstimate
    leak_ec_real: int
    leak_ec_teorico: float
    discrepancias_tras_cascade: int
    autenticacion_ok: bool
    abortado: bool
    razon: str
    longitud_clave_final: int
    tasa_asintotica_bits_por_pulso: float
    tasa_empirica_bits_por_pulso: float
    detector_click_rate: float
    dark_click_rate: float
    double_click_rate: float
    clave_final_alice: np.ndarray
    clave_final_bob: np.ndarray

    @property
    def claves_coinciden(self) -> bool:
        return bool(np.array_equal(self.clave_final_alice, self.clave_final_bob))

    @property
    def qber_medido(self) -> float:
        """v5.4: sustituye al campo `expected_qber`, que en los dos
        constructores de SecurityReport se limitaba a asignar
        literalmente bit_error.value -- un duplicado sin cálculo propio
        que solo podía desincronizarse por error, nunca aportar
        información nueva."""
        return self.bit_error.value

    @property
    def f_ec_empirico(self) -> float:
        """
        f_EC realmente medido en ESTA ejecución:
            f_EC = leak_ec_real / (n_resto * h(QBER))
        Cascade real bien ajustado: ~1.05-1.20 (ver SecurityParameters
        .fec_efficiency, que es el valor DE REFERENCIA usado para
        leak_ec_teorico; este es el que se observó de verdad).
        NaN si no hay datos suficientes (p.ej. ejecución abortada antes
        de reconciliar, o QBER puntual == 0 exacto).
        """
        n_resto = self.n_tamizada - self.n_verificacion
        if n_resto <= 0 or self.bit_error.value <= 0:
            return float("nan")
        h = float(entropia_binaria(self.bit_error.value))
        if h <= 0:
            return float("nan")
        return self.leak_ec_real / (n_resto * h)


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


# ============================================================================
# 8. Punto de Entrada Principal (Ejemplo de Uso)
# ============================================================================

if __name__ == "__main__":
    # 1. Validar la integridad estadística del código
    run_statistical_tests()

    # 2. Configurar entorno de prueba
    #
    # NOTA sobre n_qubits y epsilon_pe: con epsilon_pe=1e-10 (muy exigente:
    # 1 fallo en 10 000 millones), la cota de Serfling necesita una
    # submuestra de verificación grande para ser ajustada. Con
    # n_qubits=50_000 a 15 km (el valor que traía v5.2 en este demo),
    # n_tamizada queda en ~3 200 bits y el margen de Serfling por sí solo
    # ronda el 14%, disparando un aborto por "QBER alto" EN UN CANAL SANO
    # (qber_intrinseco=0.015, sin espía) -- no es un bug del código, es
    # una consecuencia esperada de pedir tantos nueves de confianza con
    # pocas muestras. Se sube a n_qubits=300_000 para que n_tamizada
    # ronde 19 000 y el margen quede muy por debajo del umbral en un
    # canal sano; si reduces epsilon_pe (menos exigente) o n_qubits, esto
    # puede volver a pasar -- es esperable, no señal de que algo se haya
    # roto.
    entropy = EntropySource.simulation(seed=2026)
    security_params = SecurityParameters(
        epsilon_pe=1e-10,
        epsilon_pa=1e-10,
        epsilon_auth=1e-12,
        fec_efficiency=1.12,
        # tag_length ya no se fija a mano: se deriva de epsilon_auth
        # (ver SecurityParameters.tag_length_efectivo).
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
    # v5.4: run() acepta directamente un ChannelModel que no sea de fibra --
    # ya no hace falta orquestar QuantumLayer/ClassicalLayer/SecurityLayer a
    # mano como en v5.3 (comparar con el comentario que llevaba esta demo).
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