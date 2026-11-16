"""Fitness: MAE medio en validación cruzada con folds fijos y caché por fenotipo.

Implementa D6, D8, D16 y D18 (RF-05, RF-07).

- El fitness es el MAE medio (media de los MAE de cada fold) de M en la CV sobre train.
- Los folds se fijan al crear la clase y son los mismos para todos los individuos.
- Por fold se ajusta un pipeline escalado + M solo con la parte de entrenamiento del
  fold (D8); el escalado de Xt′ vive dentro del pipeline.
- Las fórmulas constantes en train no entran en Xt′ (D14, columnas efectivas).
- La caché de fitness (huella -> MAE, sin límite: cada entrada es un float) devuelve el valor
  de un fenotipo ya evaluado sin recalcularlo (D18). Es distinta de la caché LRU de vectores
  de valores de fórmulas (``max_cached_values``), que solo afecta al tiempo y a la memoria.

Esta clase solo ve (X, y) de train: nunca recibe ni lee el conjunto de validación.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .individual import ColumnEvaluator, Individual, build_matrix, phenotype_key

Folds = Sequence[tuple[np.ndarray, np.ndarray]]  # (índices de train, índices de test)


class Fitness:
    """Evaluador de individuos: ``fitness(ind)`` es el MAE medio en CV (a minimizar)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: Folds,
        model_factory: Callable[[], Any],
        rep: Any,
        max_cached_values: int = 1000,
    ) -> None:
        self.X = X  # originales de train, ya escalados (D2)
        self.y = y
        self.folds = list(folds)
        self.model_factory = model_factory  # devuelve un modelo de scikit-learn sin ajustar
        self.rep = rep
        self.evaluator = ColumnEvaluator(X, rep, max_cached_values)  # fórmulas sobre train, con constancia (D14)
        self._cache: dict[str, float] = {}
        self.evaluations = 0  # evaluaciones reales (sin contar aciertos de caché)
        self.cache_hits = 0

    def _fold_maes(self, M: np.ndarray, folds: Folds | None = None) -> list[float]:
        """MAE de cada fold, ajustando escalado + modelo solo con el train de cada fold."""
        maes = []
        for entrenar, probar in (self.folds if folds is None else folds):
            modelo = make_pipeline(StandardScaler(), self.model_factory())
            modelo.fit(M[entrenar], self.y[entrenar])
            maes.append(float(np.mean(np.abs(self.y[probar] - modelo.predict(M[probar])))))
        return maes

    def _mae_cv(self, M: np.ndarray, folds: Folds | None = None) -> float:
        """MAE medio sobre los folds (media de los MAE de cada fold)."""
        return float(np.mean(self._fold_maes(M, folds)))

    def fold_maes(self, ind: Individual, folds: Folds | None = None) -> list[float]:
        """MAE de cada fold de un individuo (por defecto, los folds de la evolución)."""
        return self._fold_maes(build_matrix(ind, self.evaluator), folds)

    def cv_mae(self, ind: Individual, folds: Folds) -> float:
        """MAE medio en CV con folds distintos de los de la evolución, sin caché (D7)."""
        M = build_matrix(ind, self.evaluator)
        if M.shape[1] == 0:
            raise ValueError("el individuo no tiene columnas efectivas; hay que repararlo antes")
        return self._mae_cv(M, folds)

    def __call__(self, ind: Individual) -> float:
        clave = phenotype_key(ind, self.evaluator)
        if clave in self._cache:
            self.cache_hits += 1
            return self._cache[clave]
        M = build_matrix(ind, self.evaluator)
        if M.shape[1] == 0:
            raise ValueError("el individuo no tiene columnas efectivas; hay que repararlo antes")
        valor = self._mae_cv(M)
        self._cache[clave] = valor
        self.evaluations += 1
        return valor
