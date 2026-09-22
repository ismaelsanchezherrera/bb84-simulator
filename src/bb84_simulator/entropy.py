"""
Fuente de entropía y generación de números aleatorios para la simulación.
"""

from __future__ import annotations

import os
import secrets
from typing import Union
import numpy as np


class EntropySource:
    """Proveedor de entropía configurable para simulaciones deterministas o aleatorias reales."""

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
        n_bytes = (n + 7) // 8
        crudo = np.frombuffer(os.urandom(n_bytes), dtype=np.uint8)
        return np.unpackbits(crudo)[:n].astype(np.int64)

    @staticmethod
    def _uniforme01_criptografico(n: int) -> np.ndarray:
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

    def poisson(self, lam: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Genera variables aleatorias de Poisson soportando entradas escalares o arrays.
        
        Límite de estabilidad: lambda <= 700 (por underflow en e^-lambda de Knuth).
        """
        lam_arr = np.atleast_1d(lam)
        
        if np.any(lam_arr < 0) or not np.all(np.isfinite(lam_arr)):
            raise ValueError(f"El parámetro lambda debe ser no negativo y finito (obtenido: {lam})")
        
        if np.any(lam_arr > 700):
            raise ValueError(f"El parámetro lambda excede el límite de estabilidad del algoritmo (lambda <= 700)")

        # Caso especial para lambda = 0
        res = np.zeros_like(lam_arr, dtype=int)
        mask_pos = lam_arr > 0
        
        if not np.any(mask_pos):
            return int(res[0]) if np.ndim(lam) == 0 else res

        # Algoritmo de Knuth para elementos con lambda > 0
        l_vals = np.exp(-lam_arr[mask_pos])
        k_vals = np.zeros(np.sum(mask_pos), dtype=int)
        p_vals = np.ones(np.sum(mask_pos), dtype=float)

        while True:
            active = p_vals > l_vals
            if not np.any(active):
                break
            k_vals[active] += 1
            p_vals[active] *= self.random(np.sum(active))

        res[mask_pos] = k_vals - 1
        return int(res[0]) if np.ndim(lam) == 0 else res

    def raw_crypto_bits(self, size: int) -> np.ndarray:
        """Genera bits aleatorios no acotados por un PRNG para seguridad ITS (Toeplitz/LHL)."""
        if self.mode == "crypto":
            # Extrae bytes directamente del sistema operativo sin semilla ni PRNG de estado fijo
            n_bytes = (size + 7) // 8
            raw_bytes = os.urandom(n_bytes)
            bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
            return bits[:size]
        else:
            return self.integers(0, 2, size=size)