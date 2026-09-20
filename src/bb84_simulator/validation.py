"""
Funciones auxiliares para la validación de parámetros y rangos.
"""

import math
from numbers import Real
from typing import Any


def validar_probabilidad(nombre: str, valor: Any) -> None:
    """Valida que `valor` sea un número real en el intervalo cerrado [0.0, 1.0]."""
    if isinstance(valor, bool) or not isinstance(valor, Real) or not (0.0 <= valor <= 1.0):
        raise ValueError(f"{nombre} debe estar en el intervalo [0, 1].")


def validar_epsilon(nombre: str, valor: Any) -> None:
    """Valida que un parámetro de seguridad epsilon esté en el intervalo (0.0, 1.0)."""
    if isinstance(valor, bool) or not isinstance(valor, Real) or not (0.0 < valor < 1.0):
        raise ValueError(f"{nombre} debe estar en el intervalo (0, 1).")


def validar_no_negativo(nombre: str, valor: Any) -> None:
    """Valida que `valor` sea un real finito mayor o igual a 0."""
    if (
        isinstance(valor, bool)
        or not isinstance(valor, Real)
        or not math.isfinite(valor)
        or valor < 0.0
    ):
        raise ValueError(f"{nombre} debe ser un real finito >= 0.")


def validar_entero_positivo(nombre: str, valor: Any) -> None:
    """Valida que un valor sea un entero estrictamente mayor que cero."""
    if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0:
        raise ValueError(f"{nombre} debe ser un entero estrictamente positivo.")