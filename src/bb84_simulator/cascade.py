"""
Algoritmo de reconciliación de errores Cascade para QKD BB84.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from bb84_simulator.entropy import EntropySource


def _paridad(bits: np.ndarray) -> int:
    """Calcula la paridad (suma mod 2) de un array de bits."""
    return int(np.sum(bits) & 1)


def error_correction_cascade(
    entropy: EntropySource,
    clave_alice: np.ndarray,
    clave_bob: np.ndarray,
    qber_estimado: float,
    n_pasadas: int = 4,
) -> tuple[np.ndarray, int, int]:
    """
    Cascade con backtracking en cola global (Brassard & Salvail, 1993).
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

    semilla_publica_cascade = entropy.random_seed_int()
    rng_publico = np.random.default_rng(semilla_publica_cascade)

    bloques_por_pasada = []
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

        for p_af in range(n_pasadas):
            b_af = int(bloque_de_idx[p_af][err_orig])
            nueva = paridades_bob[p_af][b_af] ^ 1
            paridades_bob[p_af][b_af] = nueva
            if nueva != paridades_alice[p_af][b_af]:
                enqueue(p_af, b_af)

    discrepancias = int(np.count_nonzero(clave_alice != bob_actual))
    return bob_actual, bits_revelados, discrepancias