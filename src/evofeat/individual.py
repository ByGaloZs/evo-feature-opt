"""Individuo: máscara de originales + K genomas + K interruptores.

Implementa D13 y D24 (RF-06), y la huella de D18 / complejidad de D17 (RF-07).

Un individuo codifica a la vez qué originales pasan a Xt′ (máscara), K fórmulas y qué
fórmulas pasan a Xt′ (interruptores). Gen k e interruptor k son una unidad.

- ``repair`` garantiza 1 ≤ columnas ≤ n − 1 (D13 si queda vacío, D24 si se pasa).
- ``phenotype_key`` no depende del orden de los genes (D18): usa el conjunto ordenado
  de fórmulas activas y la máscara.
- Columnas constantes: con OLS con término independiente, una columna constante no cambia
  las predicciones (es colineal con el intercepto), así que no altera el fitness; solo
  cuenta como columna en ``n_columns`` y en la complejidad. Ver nota en el hito 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .formula import evaluate, fingerprint, size

Genome = Any  # tuple[str, ...] en ambas representaciones


@dataclass(frozen=True, eq=False)
class Individual:
    """Individuo (inmutable). ``eq=False``: los arrays de NumPy no se comparan con ``==``."""

    mask: np.ndarray  # bool (n,)  originales en Xt′
    genomes: tuple[Genome, ...]  # K fórmulas (genomas de la representación)
    switches: np.ndarray  # bool (K,)  fórmulas en Xt′

    def __post_init__(self) -> None:
        if len(self.genomes) != len(self.switches):
            raise ValueError("debe haber un interruptor por genoma")

    @property
    def n_columns(self) -> int:
        """Columnas de Xt′: originales activos + fórmulas activas."""
        return int(self.mask.sum() + self.switches.sum())


def repair(ind: Individual, n: int, rng: np.random.Generator) -> Individual:
    """Deja el individuo con entre 1 y n − 1 columnas (D13, D24).

    - Si no hay ninguna, activa un bit inactivo al azar (de la máscara o de los interruptores).
    - Si hay más de n − 1, desactiva bits activos al azar hasta cumplir el límite.
    """
    if n < 2:
        raise ValueError("con n < 2 no hay forma de cumplir 1 ≤ columnas ≤ n − 1")
    bits = np.concatenate([ind.mask, ind.switches]).astype(bool)  # copia
    activos = int(bits.sum())
    if activos == 0:
        bits[int(rng.integers(len(bits)))] = True
    elif activos > n - 1:
        a_apagar = rng.choice(np.flatnonzero(bits), size=activos - (n - 1), replace=False)
        bits[a_apagar] = False
    return Individual(mask=bits[:n].copy(), genomes=ind.genomes, switches=bits[n:].copy())


def active_formulas(ind: Individual, rep: Any) -> list[tuple[str, ...]]:
    """Fórmulas (preorden) de los genomas con interruptor activo, en el orden de los genes."""
    return [rep.to_formula(g) for g, s in zip(ind.genomes, ind.switches) if s]


def build_matrix(ind: Individual, X: np.ndarray, rep: Any) -> np.ndarray:
    """Xt′ = originales activos (en su orden) + fórmulas activas (en el orden de los genes).

    ``X`` son los originales ya escalados (D2); las fórmulas pueden usar todos los
    originales, aunque la máscara los quite de Xt′.
    """
    columnas = [X[:, ind.mask]]
    columnas += [evaluate(f, X).reshape(-1, 1) for f in active_formulas(ind, rep)]
    return np.hstack(columnas)


def complexity(ind: Individual, rep: Any) -> int:
    """Complejidad (D17): columnas de Xt′ + nodos de las fórmulas activas."""
    return ind.n_columns + sum(size(f) for f in active_formulas(ind, rep))


def phenotype_key(ind: Individual, rep: Any) -> str:
    """Huella para la caché de fitness (D18), independiente del orden de los genes.

    Es la máscara más el conjunto ordenado de fórmulas activas. Se usa un conjunto:
    dos genes con la misma fórmula dan las mismas predicciones que uno solo.
    """
    mascara = "".join("1" if b else "0" for b in ind.mask)
    formulas = sorted({fingerprint(f) for f in active_formulas(ind, rep)})
    return mascara + "|" + "|".join(formulas)
