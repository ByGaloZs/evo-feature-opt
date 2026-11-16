"""Representación común de una fórmula: tupla de símbolos en preorden.

Implementa D14, D15 y §15.1 (RF-04, RNF-04): evaluación vectorizada, profundidad,
tamaño, impresión infija y huella. Árboles y GEP comparten este código (D25).

Convenciones:
- Una fórmula es ``tuple[str, ...]`` en preorden, p. ej. ``("mul", "x3", "log", "x1")``.
- Los terminales son ``x<i>`` con ``i`` el índice de columna en base 0
  (``x0`` es la primera columna de X); no hay constantes (D15).
- La raíz tiene profundidad 0 (D13); un terminal suelto tiene profundidad 0 y tamaño 1.
- Evaluación con una pila recorriendo la fórmula al revés, sin recursión (§15.1).
"""

from __future__ import annotations

import numpy as np

from .primitives import PRIMITIVES

Formula = tuple[str, ...]


def is_terminal(s: str) -> bool:
    """True si el símbolo es un terminal ``x<i>``."""
    return len(s) > 1 and s[0] == "x" and s[1:].isdigit()


def terminal_index(s: str) -> int:
    """Índice de columna (base 0) de un terminal ``x<i>``."""
    if not is_terminal(s):
        raise ValueError(f"símbolo '{s}' no es un terminal")
    return int(s[1:])


def is_valid(f: Formula) -> bool:
    """True si ``f`` es un preorden bien formado (símbolos conocidos y aridades que cuadran)."""
    if len(f) == 0:
        return False
    pendientes = 1  # subárboles que faltan por completar
    for i, s in enumerate(f):
        if pendientes == 0:  # sobran símbolos tras completar el árbol
            return False
        if s in PRIMITIVES:
            pendientes += PRIMITIVES[s][0] - 1
        elif is_terminal(s):
            pendientes -= 1
        else:
            return False
    return pendientes == 0


def _comprobar(f: Formula) -> None:
    if not is_valid(f):
        raise ValueError(f"fórmula mal formada: {f}")


def evaluate(f: Formula, X: np.ndarray) -> np.ndarray:
    """Evalúa ``f`` sobre X (m, n) y devuelve un vector (m,) finito.

    Vectorizada: la pila guarda columnas enteras, no filas. Para X finita el
    resultado es siempre finito gracias a las primitivas protegidas.
    """
    _comprobar(f)
    pila: list[np.ndarray] = []
    for s in reversed(f):
        if s in PRIMITIVES:
            aridad, func = PRIMITIVES[s]
            # al recorrer al revés, el operando izquierdo queda arriba de la pila
            args = [pila.pop() for _ in range(aridad)]
            pila.append(func(*args))
        else:
            pila.append(X[:, terminal_index(s)].astype(np.float64, copy=True))
    return pila[0]


def depth(f: Formula) -> int:
    """Profundidad del árbol (raíz = 0; un terminal solo = 0)."""
    _comprobar(f)
    pila: list[int] = []
    for s in reversed(f):
        if s in PRIMITIVES:
            aridad = PRIMITIVES[s][0]
            hijos = [pila.pop() for _ in range(aridad)]
            pila.append(1 + max(hijos))
        else:
            pila.append(0)
    return pila[0]


def size(f: Formula) -> int:
    """Número de nodos (símbolos) de la fórmula."""
    _comprobar(f)
    return len(f)


_BINARIOS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def to_infix(f: Formula, names: list[str]) -> str:
    """Notación infija legible con los nombres reales de las columnas (RNF-04).

    Binarias entre paréntesis; ``sqrt(|a|)`` es Q y ``log(1 + |a|)`` es L.
    """
    _comprobar(f)
    pila: list[str] = []
    for s in reversed(f):
        if s in _BINARIOS:
            a, b = pila.pop(), pila.pop()
            pila.append(f"({a} {_BINARIOS[s]} {b})")
        elif s == "sqrt":
            pila.append(f"sqrt(|{pila.pop()}|)")
        elif s == "log":
            pila.append(f"log(1 + |{pila.pop()}|)")
        else:
            pila.append(names[terminal_index(s)])
    return pila[0]


def fingerprint(f: Formula) -> str:
    """Huella de la fórmula (para la caché de fitness, D18)."""
    return " ".join(f)
