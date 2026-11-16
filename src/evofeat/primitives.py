"""Funciones protegidas del conjunto de funciones.

Implementa D14 (RF-04): +, −, ×, ÷ protegida, Q(x) = √|x| y L(x) = log(1 + |x|).

- ÷ devuelve 1 si |denominador| < 1e-6 (convención de Koza, §15.5).
- Q y L usan |x|, así que están definidas para todo real.
- Con entradas finitas, todas las primitivas devuelven valores finitos: los
  resultados de +, −, × y ÷ se acotan a ±LIMITE para que el desbordamiento
  (p. ej. 1e200 · 1e200) no produzca ``inf`` ni ``nan`` al encadenar operaciones.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import numpy as np

UMBRAL_DIV = 1e-6  # |denominador| < UMBRAL_DIV -> la división devuelve 1
LIMITE = 1e150  # cota de seguridad contra desbordamiento (ver docstring del módulo)


def _seguro(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Silencia avisos de desbordamiento y acota el resultado a ±LIMITE."""

    @wraps(f)
    def envuelta(*args: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.clip(f(*args), -LIMITE, LIMITE)

    return envuelta


@_seguro
def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


@_seguro
def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


@_seguro
def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


@_seguro
def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """División protegida: 1 donde |b| < 1e-6."""
    protegido = np.abs(b) < UMBRAL_DIV
    return np.where(protegido, 1.0, a / np.where(protegido, 1.0, b))


@_seguro
def sqrt(a: np.ndarray) -> np.ndarray:
    """Q(x) = √|x|."""
    return np.sqrt(np.abs(a))


@_seguro
def log(a: np.ndarray) -> np.ndarray:
    """L(x) = log(1 + |x|)."""
    return np.log1p(np.abs(a))


# nombre -> (aridad, función)
PRIMITIVES: dict[str, tuple[int, Callable[..., np.ndarray]]] = {
    "add": (2, add),
    "sub": (2, sub),
    "mul": (2, mul),
    "div": (2, div),
    "sqrt": (1, sqrt),
    "log": (1, log),
}

FUNCTION_NAMES: tuple[str, ...] = tuple(PRIMITIVES)


def arity(name: str) -> int:
    """Aridad de una función (KeyError si no existe)."""
    return PRIMITIVES[name][0]
