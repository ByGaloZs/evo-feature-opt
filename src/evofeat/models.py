"""Fábrica de modelos: M principal (OLS) y modelos secundarios con hiperparámetros fijos.

Implementa D10, D11 y §15.5 (RF-09).

- OLS: sin hiperparámetros (D10).
- Ridge: α = 1,0. KNN: k = 5, pesos uniformes. Random Forest: 200 árboles, el resto por
  defecto, ``random_state = semilla`` y un solo hilo (§15.5). Valores por defecto de
  scikit-learn salvo los indicados; no se ajustan para que la comparación sea de datos.
- ``make_pipeline_model`` añade el escalado de Xt′ (D2, después de la síntesis) dentro del
  pipeline, de modo que se ajusta solo con el conjunto de entrenamiento (D8).
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SECONDARY_MODELS: tuple[str, ...] = ("ridge", "knn", "rf")
MODEL_NAMES: tuple[str, ...] = ("ols",) + SECONDARY_MODELS


def make_model(name: str, seed: int = 0) -> Any:
    """Modelo sin ajustar y sin escalado, con los hiperparámetros fijos del §15.5."""
    if name == "ols":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=5, weights="uniform")
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"modelo desconocido '{name}'; disponibles: {list(MODEL_NAMES)}")


def make_pipeline_model(name: str, seed: int = 0) -> Pipeline:
    """Escalado (ajustado solo con train) + modelo, listo para ``fit`` y ``predict``."""
    return make_pipeline(StandardScaler(), make_model(name, seed))
