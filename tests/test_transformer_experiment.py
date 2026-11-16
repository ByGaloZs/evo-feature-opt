"""Tests del hito 7: transformador, configuración, experimento y regla anti-fuga (RNF-02)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from evofeat import data
from evofeat.experiment import config_hash, ruta_resultado, run_and_save, run_one
from evofeat.formula import evaluate
from evofeat.transformer import EvoFeatureTransformer, RunConfig, load_config

RAIZ = Path(__file__).resolve().parents[1]

# presupuesto mínimo para que las pruebas sean rápidas
PEQUENA = dict(pop_size=20, generations=3, hof_size=5, repeats=1)


def _cfg(**kw) -> RunConfig:
    return RunConfig(**{"name": "test", **PEQUENA, **kw})


@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * [1, 10, 100, 1, 3] + [0, 5, -50, 2, 0]
    y = X[:, 0] * X[:, 1] / 10 + X[:, 2] / 50 + rng.normal(size=200)
    return X, y


# --- RunConfig ---------------------------------------------------------------------------

TOML = """
[experiment]
name = "prueba"
datasets = ["diabetes"]
configuration = "A"
representation = "gep"
seeds = { start = 5, count = 3 }

[representation]
K = 4
max_depth = 2
functions = ["add", "mul", "sqrt"]

[evolution]
pop_size = 50
generations = 10
max_columns = "n-1"

[gep]
p_point = 0.05

[fitness]
model = "ols"
epsilon = 0.01

[final_selection]
hof_size = 7
repeats = 2

[datasets.diabetes]
epsilon = 1.26

[datasets.california]
epsilon = 0.0030
"""


def test_config_desde_toml(tmp_path):
    ruta = tmp_path / "x.toml"
    ruta.write_text(TOML)
    c = load_config(str(ruta))
    assert (c.name, c.configuration, c.representation) == ("prueba", "A", "gep")
    assert (c.seed_start, c.seed_count, c.K, c.max_depth) == (5, 3, 4, 2)
    assert c.functions == ("add", "mul", "sqrt") and c.datasets == ("diabetes",)
    assert (c.pop_size, c.generations, c.p_point, c.epsilon, c.hof_size, c.repeats) == (50, 10, 0.05, 0.01, 7, 2)
    assert c.head == 3  # gep_head deducido de max_depth = 2 (D20)
    # ε por dataset (D17): depende de la escala del objetivo
    assert c.epsilon_for("diabetes") == 1.26 and c.epsilon_for("california") == 0.003
    assert c.for_dataset("diabetes").epsilon == 1.26 and c.epsilon == 0.01  # el original no cambia
    assert c.epsilon_for("otro") == 0.01  # sin sección propia, usa el ε por defecto
    assert len(config_hash(ruta)) == 64


@pytest.mark.parametrize("profundidad,cabeza", [(2, 3), (3, 7), (4, 15)])
def test_cabeza_gep_deducida_de_la_profundidad(profundidad, cabeza):
    assert RunConfig(max_depth=profundidad).head == cabeza
    assert RunConfig(max_depth=profundidad, gep_head=5).head == 5


def test_config_rechaza_claves_y_valores_desconocidos():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"evolution": {"pop_sise": 10}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"otra": {}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"evolution": {"max_columns": "n"}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"datasets": {"diabetes": {"epsilon": 1.0, "K": 3}}})
    with pytest.raises(ValueError):
        RunConfig(configuration="Z")
    with pytest.raises(ValueError):
        RunConfig(representation="otra")


# --- transformador -----------------------------------------------------------------------

@pytest.mark.parametrize("rep", ["tree", "gep"])
def test_fit_transform_basico(sintetico, rep):
    X, y = sintetico
    tf = EvoFeatureTransformer(_cfg(representation=rep), seed=1).fit(X, y)
    Z = tf.transform(X)
    assert Z.shape == (len(X), tf.n_columns) and np.isfinite(Z).all()
    assert 1 <= tf.n_columns <= X.shape[1] - 1  # D24
    assert len(tf.kept_originals_) + len(tf.formulas_) == tf.n_columns
    assert len(tf.history_) == 4  # generación 0 + 3
    assert tf.individuals_generated_ == 20 + 3 * 18  # 100 + 100·98 con el presupuesto completo
    assert tf.distinct_evaluations_ <= tf.individuals_generated_
    assert tf.distinct_evaluations_ + tf.cache_hits_ == tf.individuals_generated_


def test_transform_no_reajusta_nada(sintetico):
    X, y = sintetico
    tf = EvoFeatureTransformer(_cfg(), seed=2).fit(X, y)
    Z1 = tf.transform(X)
    rng = np.random.default_rng(5)
    otros = rng.normal(size=(30, 5)) * 1000.0  # datos muy distintos: transform no debe cambiar estado
    tf.transform(otros)
    np.testing.assert_array_equal(tf.transform(X), Z1)
    np.testing.assert_array_equal(tf.transform(X[:50]), Z1[:50])  # fila a fila, sin depender del lote


def test_transform_antes_de_fit_falla():
    with pytest.raises(RuntimeError):
        EvoFeatureTransformer(_cfg(), 0).transform(np.zeros((2, 3)))


def test_misma_semilla_mismo_transformador(sintetico):
    X, y = sintetico
    a = EvoFeatureTransformer(_cfg(), seed=3).fit(X, y)
    b = EvoFeatureTransformer(_cfg(), seed=3).fit(X, y)
    c = EvoFeatureTransformer(_cfg(), seed=4).fit(X, y)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    assert a.formulas_ == b.formulas_ and a.cv_mae_best_ == b.cv_mae_best_
    assert a.history_ != c.history_


def test_nombres_de_columnas_en_las_formulas(sintetico):
    X, y = sintetico
    nombres = ["a", "b", "c", "d", "e"]
    tf = EvoFeatureTransformer(_cfg(generations=6), seed=1).fit(X, y, feature_names=nombres)
    assert set(tf.kept_originals_) <= set(nombres)
    for f in tf.formulas_:
        assert not re.search(r"\bx\d\b", f)  # nombres reales, no x0, x1…
    with pytest.raises(ValueError):
        EvoFeatureTransformer(_cfg(), 0).fit(X, y, feature_names=["a"])


def test_baseline_devuelve_los_originales_escalados(sintetico):
    X, y = sintetico
    tf = EvoFeatureTransformer(_cfg(configuration="baseline"), seed=0).fit(X, y)
    Z = tf.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=0), 1.0)
    assert tf.n_columns == X.shape[1] and tf.formulas_ == [] and tf.history_ == []
    assert np.isfinite(tf.cv_mae_best_)


def test_imputacion_por_mediana_ajustada_en_train(sintetico):
    X, y = sintetico
    Xn = X.copy()
    Xn[:10, 1] = np.nan
    tf = EvoFeatureTransformer(_cfg(configuration="baseline"), seed=0).fit(Xn, y)
    nuevo = X[:3].copy()
    nuevo[:, 1] = np.nan
    Z = tf.transform(nuevo)
    esperado = (np.nanmedian(Xn[:, 1]) - tf.scaler_.mean_[1]) / tf.scaler_.scale_[1]
    np.testing.assert_allclose(Z[:, 1], esperado)


def test_las_columnas_constantes_no_entran_en_xt_prima():
    """Una fórmula constante en train no aparece en Xt′ ni en transform sobre datos nuevos."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(150, 4))
    X[:, 3] = 7.0  # columna constante: cualquier fórmula solo de x3 es constante
    y = X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=150)
    tf = EvoFeatureTransformer(_cfg(generations=8), seed=0).fit(X, y)
    nuevos = rng.normal(size=(40, 4))
    nuevos[:, 3] = 7.0
    Z = tf.transform(nuevos)
    assert Z.shape[1] == tf.n_columns
    Zt = tf.transform(X)
    n_orig = len(tf.kept_originals_)
    # las columnas sintetizadas efectivas no son constantes en train
    for j in range(n_orig, Zt.shape[1]):
        assert np.std(Zt[:, j]) > 1e-8


def test_reseleccion_usa_particiones_nuevas(sintetico):
    X, y = sintetico
    tf = EvoFeatureTransformer(_cfg(generations=5), seed=1).fit(X, y)
    cvs = [h["cv_mae_new_folds"] for h in tf.hall_of_fame_]
    assert cvs == sorted(cvs)  # el ganador es el de menor MAE en los folds nuevos
    assert tf.cv_mae_best_ == cvs[0]
    assert 1 <= len(tf.hall_of_fame_) <= 5


# --- experimento -------------------------------------------------------------------------------

CAMPOS = {
    "meta", "baseline_mae", "mae", "relative_improvement", "cv_mae_best", "n_columns",
    "n_originals_kept", "n_synthesized", "kept_originals", "formulas", "secondary", "history",
    "evaluations", "cache_hits", "seconds", "individuals_generated", "distinct_evaluations",
    "constant_formulas_dropped",
}


def test_ejecucion_completa_en_diabetes_con_presupuesto_pequeno(tmp_path):
    cfg = _cfg(name="smoke")
    reg = run_one(cfg, "diabetes", seed=0, hash_toml="abc")
    assert CAMPOS <= set(reg)
    assert {"experiment", "dataset", "configuration", "representation", "seed", "config_hash",
            "python", "libraries", "date"} <= set(reg["meta"])
    assert reg["meta"]["dataset"] == "diabetes" and reg["meta"]["config_hash"] == "abc"
    assert reg["meta"]["epsilon"] == 0.0
    assert set(reg["secondary"]) == {"ridge", "knn", "rf"}
    for v in reg["secondary"].values():
        assert np.isfinite(v["baseline_mae"]) and np.isfinite(v["mae"])
    assert np.isfinite(reg["baseline_mae"]) and np.isfinite(reg["mae"])
    assert reg["relative_improvement"] == pytest.approx((reg["baseline_mae"] - reg["mae"]) / reg["baseline_mae"])
    assert reg["n_columns"] == reg["n_originals_kept"] + reg["n_synthesized"] == len(reg["kept_originals"]) + len(reg["formulas"])
    assert 1 <= reg["n_columns"] <= 9  # Diabetes: n − 1 = 9
    assert reg["individuals_generated"] == 20 + 3 * 18
    assert reg["distinct_evaluations"] == reg["evaluations"]
    assert reg["distinct_evaluations"] + reg["cache_hits"] == reg["individuals_generated"]
    assert len(reg["history"]) == 4 and reg["seconds"] >= reg["seconds_fit"] > 0
    json.dumps(reg, allow_nan=False)  # serializable sin NaN


def test_experimento_usa_el_epsilon_del_dataset():
    cfg = _cfg(name="eps", epsilon_by_dataset={"diabetes": 1.26})
    assert run_one(cfg, "diabetes", seed=0)["meta"]["epsilon"] == 1.26


def test_guardado_atomico_y_no_repite(tmp_path):
    cfg = _cfg(name="smoke")
    ruta, ejecutada = run_and_save(cfg, "diabetes", 1, "h", dir_runs=tmp_path)
    assert ejecutada and ruta == ruta_resultado(cfg, "diabetes", 1, tmp_path)
    assert ruta == tmp_path / "smoke" / "diabetes" / "seed_1.json"
    datos = json.loads(ruta.read_text())
    assert CAMPOS <= set(datos)
    antes = ruta.read_bytes()
    ruta2, ejecutada2 = run_and_save(cfg, "diabetes", 1, "h", dir_runs=tmp_path)
    assert ruta2 == ruta and not ejecutada2 and ruta.read_bytes() == antes
    assert not list(tmp_path.rglob("*.tmp"))


def test_ejecucion_baseline_no_cambia_el_error(tmp_path):
    reg = run_one(_cfg(name="b", configuration="baseline"), "diabetes", seed=0)
    assert reg["mae"] == pytest.approx(reg["baseline_mae"], rel=1e-9)
    assert reg["n_synthesized"] == 0 and reg["n_columns"] == 10 and reg["history"] == []
    assert reg["evolution_best_fitness"] is None  # sin evolución


def test_transform_sobre_validacion_es_finito_y_sin_columna_constante():
    cfg = _cfg(generations=5)
    tr, va = data.load_train("diabetes"), data.load_validation("diabetes")
    tf = EvoFeatureTransformer(cfg, 0).fit(tr.X, tr.y, feature_names=tr.nombres)
    Zv = tf.transform(va.X)
    assert np.isfinite(Zv).all() and Zv.shape == (va.X.shape[0], tf.n_columns)


# --- regla anti-fuga (RNF-02) ----------------------------------------------------------------------

def test_solo_experiment_py_llama_a_load_validation():
    permitidos = {"data.py", "experiment.py"}  # data.py la define; experiment.py la llama
    fuentes = list((RAIZ / "src" / "evofeat").glob("*.py")) + list((RAIZ / "scripts").glob("*.py"))
    con_llamada = {p.name for p in fuentes if "load_validation" in p.read_text()}
    assert con_llamada == permitidos
    # data.py solo la define; la única llamada fuera de su módulo está en experiment.py
    llamadas = [p.name for p in fuentes if re.search(r"load_validation\s*\(", p.read_text()) and p.name != "data.py"]
    assert llamadas == ["experiment.py"]
    assert len(re.findall(r"load_validation\s*\(", (RAIZ / "src/evofeat/experiment.py").read_text())) == 1


def test_ningun_otro_modulo_importa_data():
    for p in (RAIZ / "src" / "evofeat").glob("*.py"):
        if p.name in {"data.py", "experiment.py", "__init__.py"}:
            continue
        texto = p.read_text()
        assert not re.search(r"^\s*(from \. import data|from \.data import|import evofeat\.data|from evofeat import data)", texto, re.M), p.name
