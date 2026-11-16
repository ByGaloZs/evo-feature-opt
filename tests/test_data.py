"""Tests del hito 1: carga de datos y particiones (D1, D4, RF-01, RF-02)."""

from __future__ import annotations

import numpy as np
import pytest

from evofeat import data
from evofeat.data import DATASETS, load_raw, make_splits, resumen


def _filas_ordenadas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    M = np.column_stack([X, y])
    return M[np.lexsort(M.T[::-1])]


@pytest.fixture(params=sorted(DATASETS))
def dataset(request) -> str:
    return request.param


@pytest.fixture
def splits(dataset, tmp_path, monkeypatch):
    """Particiona en un directorio temporal y redirige data.py a él."""
    assert make_splits(dataset, tmp_path) is True
    monkeypatch.setattr(data, "DIR_SPLITS", tmp_path)
    return tmp_path


def test_particiones_disjuntas_y_cubren_todo(dataset, splits):
    raw = load_raw(dataset)
    tr, va = data.load_train(dataset), data.load_validation(dataset)
    assert tr.X.shape[0] + va.X.shape[0] == raw.X.shape[0]
    union = _filas_ordenadas(np.vstack([tr.X, va.X]), np.concatenate([tr.y, va.y]))
    original = _filas_ordenadas(raw.X, raw.y)
    # mismo multiconjunto de filas: sin pérdidas ni repeticiones entre partes
    np.testing.assert_array_equal(union, original)


def test_proporcion_80_20(dataset, splits):
    raw = load_raw(dataset)
    va = data.load_validation(dataset)
    assert va.X.shape[0] == round(0.2 * raw.X.shape[0])


def test_misma_semilla_mismos_archivos(dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_splits(dataset, a)
    make_splits(dataset, b)
    for parte in ("train", "val"):
        assert (a / f"{dataset}_{parte}.csv").read_bytes() == (b / f"{dataset}_{parte}.csv").read_bytes()


def test_semilla_distinta_cambia_particion(dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_splits(dataset, a, semilla=1)
    make_splits(dataset, b, semilla=2)
    assert (a / f"{dataset}_val.csv").read_bytes() != (b / f"{dataset}_val.csv").read_bytes()


def test_si_existe_no_se_regenera(dataset, tmp_path):
    assert make_splits(dataset, tmp_path) is True
    antes = (tmp_path / f"{dataset}_val.csv").read_bytes()
    assert make_splits(dataset, tmp_path, semilla=999) is False
    assert (tmp_path / f"{dataset}_val.csv").read_bytes() == antes


def test_objetivo_no_aparece_en_X(dataset, splits):
    _, objetivo = DATASETS[dataset]
    for d in (load_raw(dataset), data.load_train(dataset), data.load_validation(dataset)):
        assert objetivo not in d.nombres
        assert d.X.shape[1] == len(d.nombres)
        assert d.y.shape == (d.X.shape[0],)


def test_todo_numerico_sin_faltantes(dataset):
    r = resumen(dataset)
    assert r["faltantes_X"] == 0 and r["faltantes_y"] == 0
    d = load_raw(dataset)
    assert np.isfinite(d.X).all() and np.isfinite(d.y).all()


def test_resumen_coincide_con_tabla_del_diseno():
    d, c = resumen("diabetes"), resumen("california")
    assert (d["filas"], d["atributos"]) == (442, 10)
    assert (c["filas"], c["atributos"]) == (20640, 8)


def test_columna_objetivo_inexistente_falla(tmp_path, monkeypatch):
    monkeypatch.setitem(DATASETS, "falso", ("Diabetes.csv", "no_existe"))
    with pytest.raises(ValueError, match="no_existe"):
        make_splits("falso", tmp_path)


def test_dataset_desconocido():
    with pytest.raises(KeyError):
        make_splits("nada")
