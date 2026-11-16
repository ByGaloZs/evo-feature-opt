"""Individuo: máscara de originales + K genomas + K interruptores.

Implementa D13, D14 y D24 (RF-06), y la huella de D18 / complejidad de D17 (RF-07).

Un individuo codifica a la vez qué originales pasan a Xt′ (máscara), K fórmulas y qué
fórmulas pasan a Xt′ (interruptores). Gen k e interruptor k son una unidad.

Columnas efectivas (D14): una fórmula es constante si su desviación típica sobre todo
Xt (train) es < 1e-10 · (1 + |media|). Una fórmula constante no entra en Xt′ aunque su
interruptor esté activo: el interruptor efectivo es ``interruptor AND no constante``. El
genotipo no se toca, así que una mutación posterior puede volver útil ese gen.
``n_columns``, ``complexity``, ``repair`` (D13, D24) y ``phenotype_key`` (D18) usan las
columnas efectivas. La constancia se mide una vez por fórmula sobre train y se guarda en
``ColumnEvaluator``; nunca usa validación.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .formula import Formula, evaluate, fingerprint, size

Genome = Any  # tuple[str, ...] en ambas representaciones

TOL_CONSTANTE = 1e-10  # std < TOL · (1 + |media|) => constante


class ColumnEvaluator:
    """Evalúa fórmulas sobre los originales de train (ya escalados) con caché.

    Guarda por fórmula (clave: su huella) si es constante, sin límite, y sus valores en
    una caché LRU de ``max_cached`` entradas, para acotar la memoria (en California cada
    columna ocupa ≈ 130 kB). Si se expulsan los valores, se recalculan al pedirlos.
    """

    def __init__(self, X: np.ndarray, rep: Any, max_cached: int = 1000) -> None:
        self.X = X
        self.rep = rep
        self.max_cached = max_cached
        self._constante: dict[str, bool] = {}
        self._valores: OrderedDict[str, np.ndarray] = OrderedDict()

    def values(self, f: Formula) -> np.ndarray:
        """Valores de la fórmula sobre todo Xt (solo lectura)."""
        clave = fingerprint(f)
        if clave in self._valores:
            self._valores.move_to_end(clave)
            return self._valores[clave]
        v = evaluate(f, self.X)
        v.setflags(write=False)
        if clave not in self._constante:
            self._constante[clave] = bool(np.std(v) < TOL_CONSTANTE * (1.0 + abs(float(np.mean(v)))))
        self._valores[clave] = v
        if len(self._valores) > self.max_cached:
            self._valores.popitem(last=False)
        return v

    def is_constant(self, f: Formula) -> bool:
        clave = fingerprint(f)
        if clave not in self._constante:
            self.values(f)
        return self._constante[clave]


@dataclass(frozen=True, eq=False)
class Individual:
    """Individuo (inmutable). ``eq=False``: los arrays de NumPy no se comparan con ``==``."""

    mask: np.ndarray  # bool (n,)  originales en Xt′
    genomes: tuple[Genome, ...]  # K fórmulas (genomas de la representación)
    switches: np.ndarray  # bool (K,)  fórmulas en Xt′ (antes de quitar constantes)

    def __post_init__(self) -> None:
        if len(self.genomes) != len(self.switches):
            raise ValueError("debe haber un interruptor por genoma")


def effective_switches(ind: Individual, ev: ColumnEvaluator) -> np.ndarray:
    """Interruptor efectivo = interruptor AND fórmula no constante en train."""
    return np.array(
        [bool(s) and not ev.is_constant(ev.rep.to_formula(g)) for g, s in zip(ind.genomes, ind.switches)],
        dtype=bool,
    )


def active_formulas(ind: Individual, ev: ColumnEvaluator) -> list[Formula]:
    """Fórmulas efectivas (activas y no constantes), en el orden de los genes."""
    return [ev.rep.to_formula(g) for g, e in zip(ind.genomes, effective_switches(ind, ev)) if e]


def dropped_constants(ind: Individual, ev: ColumnEvaluator) -> list[Formula]:
    """Fórmulas con interruptor activo descartadas por ser constantes en train."""
    return [
        ev.rep.to_formula(g)
        for g, s in zip(ind.genomes, ind.switches)
        if s and ev.is_constant(ev.rep.to_formula(g))
    ]


def n_columns(ind: Individual, ev: ColumnEvaluator) -> int:
    """Columnas efectivas de Xt′: originales activos + fórmulas efectivas."""
    return int(ind.mask.sum() + effective_switches(ind, ev).sum())


def repair(ind: Individual, n: int, rng: np.random.Generator, ev: ColumnEvaluator) -> Individual:
    """Deja el individuo con entre 1 y n − 1 columnas efectivas (D13, D24).

    - Si no queda ninguna, activa un original inactivo al azar.
    - Si hay más de n − 1, apaga al azar bits que aporten columna efectiva hasta cumplir
      el límite. Los interruptores de fórmulas constantes no se tocan.
    """
    if n < 2:
        raise ValueError("con n < 2 no hay forma de cumplir 1 ≤ columnas ≤ n − 1")
    mask, switches = ind.mask.astype(bool).copy(), ind.switches.astype(bool).copy()
    efectivos = effective_switches(ind, ev)
    total = int(mask.sum() + efectivos.sum())
    if total == 0:
        mask[int(rng.choice(np.flatnonzero(~mask)))] = True
    elif total > n - 1:
        aportan = np.concatenate([mask, efectivos])
        for b in rng.choice(np.flatnonzero(aportan), size=total - (n - 1), replace=False):
            if b < n:
                mask[b] = False
            else:
                switches[b - n] = False
    return Individual(mask=mask, genomes=ind.genomes, switches=switches)


def transform_matrix(mask: np.ndarray, formulas: list[Formula], X: np.ndarray) -> np.ndarray:
    """Aplica un fenotipo ya fijado (máscara + fórmulas efectivas) a cualquier X escalada.

    Es lo que usa el transformador en ``transform``: no recalcula constancia.
    """
    columnas = [X[:, mask]] + [evaluate(f, X).reshape(-1, 1) for f in formulas]
    return np.hstack(columnas)


def build_matrix(ind: Individual, ev: ColumnEvaluator) -> np.ndarray:
    """Xt′ = originales activos (en su orden) + fórmulas efectivas (en el orden de los genes).

    Las fórmulas pueden usar todos los originales, aunque la máscara los quite de Xt′.
    """
    columnas = [ev.X[:, ind.mask]]
    columnas += [ev.values(f).reshape(-1, 1) for f in active_formulas(ind, ev)]
    return np.hstack(columnas)


def complexity(ind: Individual, ev: ColumnEvaluator) -> int:
    """Complejidad (D17): columnas efectivas de Xt′ + nodos de las fórmulas efectivas."""
    return n_columns(ind, ev) + sum(size(f) for f in active_formulas(ind, ev))


def phenotype_key(ind: Individual, ev: ColumnEvaluator) -> str:
    """Huella para la caché de fitness (D18), independiente del orden de los genes.

    Es la máscara más el conjunto ordenado de fórmulas efectivas. Se usa un conjunto:
    dos genes con la misma fórmula dan las mismas predicciones que uno solo.
    """
    mascara = "".join("1" if b else "0" for b in ind.mask)
    formulas = sorted({fingerprint(f) for f in active_formulas(ind, ev)})
    return mascara + "|" + "|".join(formulas)
