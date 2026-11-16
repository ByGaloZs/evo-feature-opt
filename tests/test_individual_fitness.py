"""Tests del hito 5: individuo, reparación y fitness (D13, D24, D6, D8, D16, D18)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evofeat.fitness import Fitness
from evofeat.individual import (
    Individual,
    build_matrix,
    complexity,
    phenotype_key,
    repair,
)
from evofeat.primitives import FUNCTION_NAMES
from evofeat.repr_gep import GepRepresentation
from evofeat.repr_tree import TreeRepresentation

N, K = 6, 5


def _tree() -> TreeRepresentation:
    return TreeRepresentation(N, 3, FUNCTION_NAMES)


def _ind(mask, genomes, switches) -> Individual:
    return Individual(np.array(mask, dtype=bool), tuple(genomes), np.array(switches, dtype=bool))


def _aleatorio(rep, rng, p=0.5) -> Individual:
    return Individual(
        mask=rng.random(N) < p,
        genomes=tuple(rep.random_genome(rng) for _ in range(K)),
        switches=rng.random(K) < p,
    )


@pytest.fixture(scope="module")
def datos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, N))
    y = 3 * X[:, 0] * X[:, 1] + X[:, 2] + 0.1 * rng.normal(size=150)
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    return X, y, folds


# --- reparación --------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", [lambda: _tree(), lambda: GepRepresentation(N, 7, FUNCTION_NAMES)])
def test_reparar_10000_individuos_deja_entre_1_y_n_menos_1_columnas(rep_cls):
    rep, rng = rep_cls(), np.random.default_rng(1)
    for i in range(10_000):
        ind = _aleatorio(rep, rng, p=[0.0, 0.5, 1.0, 0.9][i % 4])  # incluye casos extremos
        rep_ind = repair(ind, N, rng)
        assert 1 <= rep_ind.n_columns <= N - 1
        assert rep_ind.mask.shape == (N,) and rep_ind.switches.shape == (K,)
        assert rep_ind.genomes == ind.genomes  # solo cambian los bits


def test_reparar_no_toca_individuos_validos_ni_muta_el_original():
    rep, rng = _tree(), np.random.default_rng(2)
    ind = _ind([1, 0, 1, 0, 0, 0], [rep.random_genome(rng)] * 2, [1, 0])
    r = repair(ind, N, rng)
    np.testing.assert_array_equal(r.mask, ind.mask)
    np.testing.assert_array_equal(r.switches, ind.switches)
    lleno = _ind([1] * N, [rep.random_genome(rng)] * 2, [1, 1])
    antes_m, antes_s = lleno.mask.copy(), lleno.switches.copy()
    repair(lleno, N, rng)
    np.testing.assert_array_equal(lleno.mask, antes_m)
    np.testing.assert_array_equal(lleno.switches, antes_s)


def test_reparar_vacio_activa_exactamente_un_bit_y_exceso_apaga_justo_lo_necesario():
    rep, rng = _tree(), np.random.default_rng(3)
    vacio = _ind([0] * N, [rep.random_genome(rng)] * K, [0] * K)
    assert repair(vacio, N, rng).n_columns == 1
    lleno = _ind([1] * N, [rep.random_genome(rng)] * K, [1] * K)
    assert repair(lleno, N, rng).n_columns == N - 1


def test_reparar_es_reproducible_y_valida_n():
    rep = _tree()
    ind = _aleatorio(rep, np.random.default_rng(4), p=1.0)
    a = repair(ind, N, np.random.default_rng(5))
    b = repair(ind, N, np.random.default_rng(5))
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.switches, b.switches)
    with pytest.raises(ValueError):
        repair(ind, 1, np.random.default_rng(0))


def test_individuo_exige_un_interruptor_por_genoma():
    with pytest.raises(ValueError):
        Individual(np.ones(N, bool), (("x0",),), np.ones(2, bool))


# --- build_matrix, complejidad ----------------------------------------------------

def test_build_matrix_orden_y_valores(datos):
    X, _, _ = datos
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    ind = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    M = build_matrix(ind, X, _tree())
    assert M.shape == (150, 3)
    np.testing.assert_allclose(M[:, 0], X[:, 0])
    np.testing.assert_allclose(M[:, 1], X[:, 2])
    np.testing.assert_allclose(M[:, 2], X[:, 0] + X[:, 1])  # f2 inactiva: no aparece


def test_formulas_usan_originales_que_la_mascara_quita(datos):
    X, _, _ = datos
    ind = _ind([0, 0, 0, 0, 0, 1], [("mul", "x0", "x1")], [1])
    M = build_matrix(ind, X, _tree())
    np.testing.assert_allclose(M[:, 1], X[:, 0] * X[:, 1])


def test_complejidad_columnas_mas_nodos_activos():
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    ind = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    assert complexity(ind, _tree()) == (2 + 1) + 3  # 3 columnas + 3 nodos de f1
    solo_orig = _ind([1, 1, 0, 0, 0, 0], [f1, f2], [0, 0])
    assert complexity(solo_orig, _tree()) == 2


# --- huella de caché ---------------------------------------------------------------

def test_huella_independiente_del_orden_de_los_genes():
    rep = _tree()
    f1, f2, f3 = ("add", "x0", "x1"), ("sqrt", "x3"), ("log", "x2")
    a = _ind([1, 0, 1, 0, 0, 0], [f1, f2, f3], [1, 0, 1])
    b = _ind([1, 0, 1, 0, 0, 0], [f3, f1, f2], [1, 1, 0])  # mismos activos {f1, f3}, otro orden
    assert phenotype_key(a, rep) == phenotype_key(b, rep)


def test_huella_distingue_mascara_y_formulas_y_ignora_inactivas_y_duplicadas():
    rep = _tree()
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    base = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    otra_mascara = _ind([1, 1, 1, 0, 0, 0], [f1, f2], [1, 0])
    otra_formula = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [0, 1])
    inactiva_distinta = _ind([1, 0, 1, 0, 0, 0], [f1, ("log", "x5")], [1, 0])
    duplicada = _ind([1, 0, 1, 0, 0, 0], [f1, f1], [1, 1])
    k = phenotype_key(base, rep)
    assert k != phenotype_key(otra_mascara, rep)
    assert k != phenotype_key(otra_formula, rep)
    assert k == phenotype_key(inactiva_distinta, rep)
    assert k == phenotype_key(duplicada, rep)


def test_huella_con_gep_usa_la_formula_decodificada():
    rep = GepRepresentation(2, 3, ("add", "mul"))  # h=3, t=4
    # ambos expresan add(x0, x1); la parte no expresada difiere
    g1 = ("add", "x0", "x1", "mul", "x0", "x0", "x1")
    g2 = ("add", "x0", "x1", "add", "x1", "x1", "x0")
    assert rep.to_formula(g1) == rep.to_formula(g2) == ("add", "x0", "x1")
    a = _ind([1, 0], [g1], [1])
    b = _ind([1, 0], [g2], [1])
    assert phenotype_key(a, rep) == phenotype_key(b, rep)


# --- fitness ---------------------------------------------------------------------------

def test_fitness_solo_originales_coincide_con_cross_val_score(datos):
    X, y, folds = datos
    rep = _tree()
    mascara = [1, 1, 0, 1, 0, 1]
    ind = _ind(mascara, [], [])
    fit = Fitness(X, y, folds, LinearRegression, rep)
    manual = -cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()),
        X[:, np.array(mascara, dtype=bool)], y, cv=folds, scoring="neg_mean_absolute_error",
    ).mean()
    assert fit(ind) == pytest.approx(manual, rel=1e-12)


def test_fitness_con_formulas_coincide_con_calculo_manual(datos):
    X, y, folds = datos
    rep = _tree()
    ind = _ind([1, 0, 0, 0, 0, 0], [("mul", "x0", "x1"), ("sqrt", "x2")], [1, 1])
    M = np.column_stack([X[:, 0], X[:, 0] * X[:, 1], np.sqrt(np.abs(X[:, 2]))])
    manual = -cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()), M, y, cv=folds,
        scoring="neg_mean_absolute_error",
    ).mean()
    assert Fitness(X, y, folds, LinearRegression, rep)(ind) == pytest.approx(manual, rel=1e-10)


def test_sintetizar_la_interaccion_mejora_el_fitness(datos):
    X, y, folds = datos
    fit = Fitness(X, y, folds, LinearRegression, _tree())
    sin = _ind([1, 1, 1, 0, 0, 0], [("mul", "x0", "x1")], [0])
    con = _ind([1, 1, 1, 0, 0, 0], [("mul", "x0", "x1")], [1])
    assert fit(con) < 0.5 * fit(sin)


def test_cache_devuelve_el_mismo_valor_y_cuenta_aciertos(datos):
    X, y, folds = datos
    rep = _tree()
    fit = Fitness(X, y, folds, LinearRegression, rep)
    f1, f2 = ("mul", "x0", "x1"), ("sqrt", "x2")
    a = _ind([1, 0, 0, 0, 0, 0], [f1, f2], [1, 1])
    v1 = fit(a)
    v2 = fit(a)
    assert v1 == v2 and (fit.evaluations, fit.cache_hits) == (1, 1)
    # mismo fenotipo con los genes en otro orden: acierto de caché y mismo valor exacto
    b = _ind([1, 0, 0, 0, 0, 0], [f2, f1], [1, 1])
    assert fit(b) == v1 and (fit.evaluations, fit.cache_hits) == (1, 2)
    # fenotipo distinto: evaluación nueva
    c = _ind([1, 1, 0, 0, 0, 0], [f2, f1], [1, 1])
    fit(c)
    assert (fit.evaluations, fit.cache_hits) == (2, 2)


def test_fitness_finito_con_formulas_extremas(datos):
    X, y, folds = datos
    rep = _tree()
    ind = _ind([1, 0, 0, 0, 0, 0], [("div", "x0", "sub", "x1", "x1"), ("mul", "mul", "x0", "x0", "mul", "x0", "x0")], [1, 1])
    assert np.isfinite(Fitness(X, y, folds, LinearRegression, rep)(ind))


def test_la_formula_constante_no_cambia_el_fitness(datos):
    """Una columna constante es colineal con el intercepto: OLS da las mismas predicciones."""
    X, y, folds = datos
    rep = _tree()
    fit = Fitness(X, y, folds, LinearRegression, rep)
    sin = _ind([1, 1, 0, 0, 0, 0], [("sub", "x0", "x0")], [0])
    con = _ind([1, 1, 0, 0, 0, 0], [("sub", "x0", "x0")], [1])
    assert fit(con) == pytest.approx(fit(sin), rel=1e-9)
