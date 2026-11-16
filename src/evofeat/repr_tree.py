"""Representación en árbol (PG): genoma = fórmula en preorden.

Implementa §8 y D13, D14, D15 (RF-06): inicialización Grow/Full y *ramped
half-and-half* (profundidades 1 a ``max_depth``), cruce de subárbol, mutación de
subárbol y puntual, y límite de profundidad (un hijo que supera ``max_depth`` se
descarta y se conserva el padre).

El intercambio de árboles completos entre individuos vive en evolution.py (§15.2).
Toda la aleatoriedad llega por un ``np.random.Generator`` pasado explícitamente.

Decisiones de detalle no fijadas por el diseño (confirmables):
- Grow elige cada nodo uniformemente entre funciones y terminales (salvo en el límite
  de profundidad, donde solo hay terminales); la raíz de Grow es siempre una función.
- Los puntos de cruce y de mutación se eligen uniformemente entre todos los nodos.
- La mutación de subárbol genera el sustituto con Grow limitado por la profundidad
  que queda disponible en ese punto, de modo que el resultado respeta ``max_depth``.
"""

from __future__ import annotations

import numpy as np

from .formula import Formula, depth, is_terminal
from .primitives import PRIMITIVES

Genome = Formula  # en la representación en árbol el genoma es la propia fórmula


def _profundidades_nodos(f: Formula) -> list[int]:
    """Profundidad de cada nodo de la fórmula (raíz = 0), en orden de preorden."""
    pila: list[int] = []  # hijos que faltan por leer de cada ancestro abierto
    profs: list[int] = []
    for s in f:
        profs.append(len(pila))
        if s in PRIMITIVES:
            pila.append(PRIMITIVES[s][0])
        else:
            while pila:
                pila[-1] -= 1
                if pila[-1] == 0:
                    pila.pop()
                else:
                    break
    return profs


def _fin_subarbol(f: Formula, i: int) -> int:
    """Índice (exclusivo) donde termina el subárbol que empieza en ``i``."""
    pendientes = 1
    j = i
    while pendientes > 0:
        s = f[j]
        pendientes += PRIMITIVES[s][0] - 1 if s in PRIMITIVES else -1
        j += 1
    return j


class TreeRepresentation:
    """Operadores de PG con árboles en preorden (interfaz ``Representation``, §15.2)."""

    def __init__(self, n_features: int, max_depth: int, functions: tuple[str, ...] | list[str]) -> None:
        if n_features < 1:
            raise ValueError("n_features debe ser ≥ 1")
        if max_depth < 1:
            raise ValueError("max_depth debe ser ≥ 1")
        desconocidas = [f for f in functions if f not in PRIMITIVES]
        if desconocidas or len(functions) == 0:
            raise ValueError(f"conjunto de funciones inválido: {list(functions)}")
        self.n_features = n_features
        self.max_depth = max_depth
        self.functions = tuple(functions)
        self.terminals = tuple(f"x{i}" for i in range(n_features))

    # ---- generación ------------------------------------------------------

    def _terminal(self, rng: np.random.Generator) -> str:
        return self.terminals[int(rng.integers(len(self.terminals)))]

    def _funcion(self, rng: np.random.Generator) -> str:
        return self.functions[int(rng.integers(len(self.functions)))]

    def _generar(self, rng: np.random.Generator, restante: int, metodo: str, raiz: bool) -> list[str]:
        """Genera un subárbol en preorden con profundidad ≤ ``restante`` (Grow o Full)."""
        if restante == 0:
            return [self._terminal(rng)]
        if metodo == "full" or raiz:
            simbolo = self._funcion(rng)
        else:  # Grow: uniforme entre funciones y terminales
            k = int(rng.integers(len(self.functions) + len(self.terminals)))
            simbolo = self.functions[k] if k < len(self.functions) else self.terminals[k - len(self.functions)]
        if simbolo not in PRIMITIVES:
            return [simbolo]
        out = [simbolo]
        for _ in range(PRIMITIVES[simbolo][0]):
            out += self._generar(rng, restante - 1, metodo, False)
        return out

    def grow(self, rng: np.random.Generator, max_depth: int | None = None) -> Genome:
        """Método Grow: profundidad ≤ max_depth, raíz siempre función."""
        d = self.max_depth if max_depth is None else max_depth
        return tuple(self._generar(rng, d, "grow", True))

    def full(self, rng: np.random.Generator, max_depth: int | None = None) -> Genome:
        """Método Full: todas las hojas a profundidad exactamente max_depth."""
        d = self.max_depth if max_depth is None else max_depth
        return tuple(self._generar(rng, d, "full", True))

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """*Ramped half-and-half*: profundidad uniforme en 1..max_depth, Grow o Full al 50 %."""
        d = int(rng.integers(1, self.max_depth + 1))
        return self.full(rng, d) if rng.random() < 0.5 else self.grow(rng, d)

    def to_formula(self, g: Genome) -> Formula:
        return g

    # ---- operadores ------------------------------------------------------

    def crossover(self, a: Genome, b: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
        """Cruce de subárbol. Un hijo que supera ``max_depth`` se sustituye por su padre."""
        i, j = int(rng.integers(len(a))), int(rng.integers(len(b)))
        ei, ej = _fin_subarbol(a, i), _fin_subarbol(b, j)
        h1 = a[:i] + b[j:ej] + a[ei:]
        h2 = b[:j] + a[i:ei] + b[ej:]
        if depth(h1) > self.max_depth:
            h1 = a
        if depth(h2) > self.max_depth:
            h2 = b
        return h1, h2

    def mutate(self, g: Genome, rng: np.random.Generator) -> Genome:
        """Mutación de subárbol o puntual (50 % cada una)."""
        if rng.random() < 0.5:
            return self._mutacion_subarbol(g, rng)
        return self._mutacion_puntual(g, rng)

    def _mutacion_subarbol(self, g: Genome, rng: np.random.Generator) -> Genome:
        i = int(rng.integers(len(g)))
        disponible = self.max_depth - _profundidades_nodos(g)[i]
        nuevo = self._generar(rng, disponible, "grow", False)
        hijo = g[:i] + tuple(nuevo) + g[_fin_subarbol(g, i):]
        return hijo if depth(hijo) <= self.max_depth else g

    def _mutacion_puntual(self, g: Genome, rng: np.random.Generator) -> Genome:
        """Cambia un nodo por otro de igual aridad (si no hay alternativa, no cambia nada)."""
        i = int(rng.integers(len(g)))
        s = g[i]
        if is_terminal(s):
            candidatos = [t for t in self.terminals if t != s]
        else:
            aridad = PRIMITIVES[s][0]
            candidatos = [f for f in self.functions if f != s and PRIMITIVES[f][0] == aridad]
        if not candidatos:
            return g
        return g[:i] + (candidatos[int(rng.integers(len(candidatos)))],) + g[i + 1:]
