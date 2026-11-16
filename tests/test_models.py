"""Tests del hito 7: fábrica de modelos (D10, D11, §15.5)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from evofeat.models import MODEL_NAMES, SECONDARY_MODELS, make_model, make_pipeline_model


def test_hiperparametros_fijos_del_diseno():
    assert isinstance(make_model("ols"), LinearRegression)
    r = make_model("ridge")
    assert isinstance(r, Ridge) and r.alpha == 1.0
    k = make_model("knn")
    assert isinstance(k, KNeighborsRegressor) and k.n_neighbors == 5 and k.weights == "uniform"
    rf = make_model("rf", seed=7)
    assert isinstance(rf, RandomForestRegressor)
    assert rf.n_estimators == 200 and rf.random_state == 7 and rf.n_jobs == 1


def test_nombres_y_error():
    assert SECONDARY_MODELS == ("ridge", "knn", "rf") and MODEL_NAMES[0] == "ols"
    with pytest.raises(ValueError):
        make_model("xgboost")


def test_pipeline_escala_y_ajusta_solo_con_train():
    rng = np.random.default_rng(0)
    Xt, Xv = rng.normal(5.0, 3.0, size=(80, 3)), rng.normal(5.0, 3.0, size=(20, 3))
    y = Xt[:, 0] + rng.normal(size=80)
    pipe = make_pipeline_model("ridge").fit(Xt, y)
    esc = pipe.steps[0][1]
    assert isinstance(esc, StandardScaler)
    np.testing.assert_allclose(esc.mean_, Xt.mean(axis=0))  # ajustado con train, no con validación
    assert np.isfinite(pipe.predict(Xv)).all()
