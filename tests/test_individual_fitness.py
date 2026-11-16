"""Tests del hito 5: individuo, reparación y fitness (D13, D14, D24, D6, D8, D16, D18)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evofeat.fitness import Fitness
from evofeat.individual import (
    ColumnEvaluator,
    Individual,
    active_formulas,
    build_matrix,
    complexity,
    dropped_constants,
    n_columns,
    phenotype_key,
    repair,
    transform_matrix,
)
from evofeat.primitives import FUNCTION_NAMES
from evofeat.repr_gep import GepRepresentation
from evofeat.repr_tree import TreeRepresentation

N, K = 6, 5
CONSTANTE = ("sub", "x0", "x0")  # x0 − x0 = 0 en cualquier dato


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


@pytest.fixture(scope="module")
def ev(datos) -> ColumnEvaluator:
    return ColumnEvaluator(datos[0], _tree())


# --- reparación --------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", [lambda: _tree(), lambda: GepRepresentation(N, 7, FUNCTION_NAMES)])
def test_reparar_10000_individuos_deja_entre_1_y_n_menos_1_columnas(datos, rep_cls):
    rep, rng = rep_cls(), np.random.default_rng(1)
    evr = ColumnEvaluator(datos[0], rep)
    for i in range(10_000):
        ind = _aleatorio(rep, rng, p=[0.0, 0.5, 1.0, 0.9][i % 4])  # incluye casos extremos
        r = repair(ind, N, rng, evr)
        assert 1 <= n_columns(r, evr) <= N - 1
        assert r.mask.shape == (N,) and r.switches.shape == (K,)
        assert r.genomes == ind.genomes  # solo cambian los bits


def test_reparar_no_toca_individuos_validos_ni_muta_el_original(ev):
    rep, rng = _tree(), np.random.default_rng(2)
    ind = _ind([1, 0, 1, 0, 0, 0], [rep.random_genome(rng)] * 2, [1, 0])
    r = repair(ind, N, rng, ev)
    np.testing.assert_array_equal(r.mask, ind.mask)
    np.testing.assert_array_equal(r.switches, ind.switches)
    lleno = _ind([1] * N, [rep.random_genome(rng)] * 2, [1, 1])
    antes_m, antes_s = lleno.mask.copy(), lleno.switches.copy()
    repair(lleno, N, rng, ev)
    np.testing.assert_array_equal(lleno.mask, antes_m)
    np.testing.assert_array_equal(lleno.switches, antes_s)


def test_reparar_vacio_activa_un_original_y_exceso_apaga_justo_lo_necesario(ev):
    rep, rng = _tree(), np.random.default_rng(3)
    vacio = _ind([0] * N, [rep.random_genome(rng)] * K, [0] * K)
    r = repair(vacio, N, rng, ev)
    assert r.mask.sum() == 1 and r.switches.sum() == 0  # activa un original
    lleno = _ind([1] * N, [rep.random_genome(rng)] * K, [1] * K)
    assert n_columns(repair(lleno, N, rng, ev), ev) == N - 1


def test_reparar_es_reproducible_y_valida_n(ev):
    rep = _tree()
    ind = _aleatorio(rep, np.random.default_rng(4), p=1.0)
    a = repair(ind, N, np.random.default_rng(5), ev)
    b = repair(ind, N, np.random.default_rng(5), ev)
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.switches, b.switches)
    with pytest.raises(ValueError):
        repair(ind, 1, np.random.default_rng(0), ev)


def test_individuo_exige_un_interruptor_por_genoma():
    with pytest.raises(ValueError):
        Individual(np.ones(N, bool), (("x0",),), np.ones(2, bool))


# --- build_matrix, complejidad ----------------------------------------------------

def test_build_matrix_orden_y_valores(datos, ev):
    X = datos[0]
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    ind = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    M = build_matrix(ind, ev)
    assert M.shape == (150, 3)
    np.testing.assert_allclose(M[:, 0], X[:, 0])
    np.testing.assert_allclose(M[:, 1], X[:, 2])
    np.testing.assert_allclose(M[:, 2], X[:, 0] + X[:, 1])  # f2 inactiva: no aparece


def test_formulas_usan_originales_que_la_mascara_quita(datos, ev):
    X = datos[0]
    ind = _ind([0, 0, 0, 0, 0, 1], [("mul", "x0", "x1")], [1])
    np.testing.assert_allclose(build_matrix(ind, ev)[:, 1], X[:, 0] * X[:, 1])


def test_build_matrix_no_permite_modificar_la_cache(datos, ev):
    ind = _ind([1, 0, 0, 0, 0, 0], [("add", "x0", "x1")], [1])
    M = build_matrix(ind, ev)
    M[:, 1] += 100.0
    np.testing.assert_allclose(build_matrix(ind, ev)[:, 1], datos[0][:, 0] + datos[0][:, 1])


def test_complejidad_columnas_mas_nodos_activos(ev):
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    ind = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    assert complexity(ind, ev) == (2 + 1) + 3  # 3 columnas + 3 nodos de f1
    assert complexity(_ind([1, 1, 0, 0, 0, 0], [f1, f2], [0, 0]), ev) == 2


# --- columnas constantes (D14) -----------------------------------------------------

def test_formula_constante_equivale_a_interruptor_apagado(datos, ev):
    f = ("add", "x0", "x1")
    con = _ind([1, 0, 1, 0, 0, 0], [f, CONSTANTE], [1, 1])
    sin = _ind([1, 0, 1, 0, 0, 0], [f, CONSTANTE], [1, 0])
    assert n_columns(con, ev) == n_columns(sin, ev) == 3
    assert complexity(con, ev) == complexity(sin, ev)
    assert phenotype_key(con, ev) == phenotype_key(sin, ev)
    np.testing.assert_array_equal(build_matrix(con, ev), build_matrix(sin, ev))
    assert dropped_constants(con, ev) == [CONSTANTE] and dropped_constants(sin, ev) == []
    # el genotipo no se toca: el interruptor sigue activo
    assert bool(con.switches[1])


def test_transform_matrix_con_formulas_efectivas_no_incluye_la_constante(datos, ev):
    """El transformador guarda las fórmulas efectivas; en validación no aparece la constante."""
    X, _, _ = datos
    rng = np.random.default_rng(20)
    X_val = rng.normal(size=(40, N))  # datos "nuevos" (stand-in de validación)
    f = ("mul", "x0", "x1")
    ind = _ind([1, 0, 0, 0, 0, 0], [f, CONSTANTE], [1, 1])
    efectivas = active_formulas(ind, ev)
    assert efectivas == [f]
    Mv = transform_matrix(ind.mask, efectivas, X_val)
    assert Mv.shape == (40, 2)
    np.testing.assert_allclose(Mv[:, 1], X_val[:, 0] * X_val[:, 1])


def test_constancia_con_umbral_relativo(ev):
    assert ev.is_constant(CONSTANTE)
    assert not ev.is_constant(("add", "x0", "x1"))
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 2))
    X[:, 1] = 1e9  # columna con media enorme: std = 0 < 1e-10·(1 + 1e9)
    evr = ColumnEvaluator(X, TreeRepresentation(2, 3, FUNCTION_NAMES))
    assert evr.is_constant(("add", "x1", "x1"))
    # una fórmula con variación pequeña pero real (std ~ 1e-6) no es constante
    X[:, 1] = rng.normal(size=100) * 1e-6
    evr2 = ColumnEvaluator(X, TreeRepresentation(2, 3, FUNCTION_NAMES))
    assert not evr2.is_constant(("add", "x1", "x1"))


def test_reparacion_cuenta_columnas_efectivas(ev):
    rng = np.random.default_rng(22)
    # solo una fórmula constante activa y ningún original: sin columnas efectivas -> activa un original
    ind = _ind([0] * N, [CONSTANTE], [1])
    r = repair(ind, N, rng, ev)
    assert r.mask.sum() == 1 and bool(r.switches[0])  # el interruptor de la constante no se toca
    assert n_columns(r, ev) == 1
    # N−1 originales + 1 fórmula útil = N columnas > N−1: se apaga una; la constante se deja
    otro = _ind([1] * (N - 1) + [0], [("add", "x0", "x1"), CONSTANTE], [1, 1])
    r2 = repair(otro, N, rng, ev)
    assert n_columns(r2, ev) == N - 1 and bool(r2.switches[1])


def test_la_constante_no_ocupa_hueco_del_limite(ev):
    rng = np.random.default_rng(23)
    # N−2 originales + fórmula útil + constante: columnas efectivas = N−1 -> ya cumple, no se repara
    ind = _ind([1] * (N - 2) + [0, 0], [("add", "x0", "x1"), CONSTANTE], [1, 1])
    r = repair(ind, N, rng, ev)
    np.testing.assert_array_equal(r.mask, ind.mask)
    np.testing.assert_array_equal(r.switches, ind.switches)


def test_constancia_se_calcula_una_vez_por_formula(datos):
    evr = ColumnEvaluator(datos[0], _tree())
    evr.is_constant(CONSTANTE)
    assert len(evr._constante) == 1
    evr.is_constant(CONSTANTE)
    evr.values(CONSTANTE)
    assert len(evr._constante) == 1


def test_cache_lru_acotada_y_recalcula(datos):
    evr = ColumnEvaluator(datos[0], _tree(), max_cached=2)
    fs = [("add", "x0", "x1"), ("mul", "x0", "x1"), ("sub", "x0", "x1")]
    primero = evr.values(fs[0]).copy()
    for f in fs[1:]:
        evr.values(f)
    assert len(evr._valores) == 2
    np.testing.assert_array_equal(evr.values(fs[0]), primero)  # recalculada tras expulsarla


# --- huella de caché ---------------------------------------------------------------

def test_huella_independiente_del_orden_de_los_genes(ev):
    f1, f2, f3 = ("add", "x0", "x1"), ("sqrt", "x3"), ("log", "x2")
    a = _ind([1, 0, 1, 0, 0, 0], [f1, f2, f3], [1, 0, 1])
    b = _ind([1, 0, 1, 0, 0, 0], [f3, f1, f2], [1, 1, 0])  # mismos activos {f1, f3}, otro orden
    assert phenotype_key(a, ev) == phenotype_key(b, ev)


def test_huella_distingue_mascara_y_formulas_y_ignora_inactivas_y_duplicadas(ev):
    f1, f2 = ("add", "x0", "x1"), ("sqrt", "x3")
    base = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [1, 0])
    otra_mascara = _ind([1, 1, 1, 0, 0, 0], [f1, f2], [1, 0])
    otra_formula = _ind([1, 0, 1, 0, 0, 0], [f1, f2], [0, 1])
    inactiva_distinta = _ind([1, 0, 1, 0, 0, 0], [f1, ("log", "x5")], [1, 0])
    duplicada = _ind([1, 0, 1, 0, 0, 0], [f1, f1], [1, 1])
    k = phenotype_key(base, ev)
    assert k != phenotype_key(otra_mascara, ev)
    assert k != phenotype_key(otra_formula, ev)
    assert k == phenotype_key(inactiva_distinta, ev)
    assert k == phenotype_key(duplicada, ev)


def test_huella_con_gep_usa_la_formula_decodificada(datos):
    rep = GepRepresentation(N, 3, ("add", "mul"))  # h=3, t=4
    evr = ColumnEvaluator(datos[0], rep)
    # ambos expresan add(x0, x1); la parte no expresada difiere
    g1 = ("add", "x0", "x1", "mul", "x0", "x0", "x1")
    g2 = ("add", "x0", "x1", "add", "x1", "x1", "x0")
    assert rep.to_formula(g1) == rep.to_formula(g2) == ("add", "x0", "x1")
    a = _ind([1, 0, 0, 0, 0, 0], [g1], [1])
    b = _ind([1, 0, 0, 0, 0, 0], [g2], [1])
    assert phenotype_key(a, evr) == phenotype_key(b, evr)


# --- fitness ---------------------------------------------------------------------------

def test_fitness_solo_originales_coincide_con_cross_val_score(datos):
    X, y, folds = datos
    mascara = [1, 1, 0, 1, 0, 1]
    ind = _ind(mascara, [], [])
    fit = Fitness(X, y, folds, LinearRegression, _tree())
    manual = -cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()),
        X[:, np.array(mascara, dtype=bool)], y, cv=folds, scoring="neg_mean_absolute_error",
    ).mean()
    assert fit(ind) == pytest.approx(manual, rel=1e-12)


def test_fitness_con_formulas_coincide_con_calculo_manual(datos):
    X, y, folds = datos
    ind = _ind([1, 0, 0, 0, 0, 0], [("mul", "x0", "x1"), ("sqrt", "x2")], [1, 1])
    M = np.column_stack([X[:, 0], X[:, 0] * X[:, 1], np.sqrt(np.abs(X[:, 2]))])
    manual = -cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()), M, y, cv=folds,
        scoring="neg_mean_absolute_error",
    ).mean()
    assert Fitness(X, y, folds, LinearRegression, _tree())(ind) == pytest.approx(manual, rel=1e-10)


def test_sintetizar_la_interaccion_mejora_el_fitness(datos):
    X, y, folds = datos
    fit = Fitness(X, y, folds, LinearRegression, _tree())
    sin = _ind([1, 1, 1, 0, 0, 0], [("mul", "x0", "x1")], [0])
    con = _ind([1, 1, 1, 0, 0, 0], [("mul", "x0", "x1")], [1])
    assert fit(con) < 0.5 * fit(sin)


def test_cache_devuelve_el_mismo_valor_y_cuenta_aciertos(datos):
    X, y, folds = datos
    fit = Fitness(X, y, folds, LinearRegression, _tree())
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
    ind = _ind([1, 0, 0, 0, 0, 0],
               [("div", "x0", "sub", "x1", "x1"), ("mul", "mul", "x0", "x0", "mul", "x0", "x0")], [1, 1])
    assert np.isfinite(Fitness(X, y, folds, LinearRegression, _tree())(ind))


def test_fitness_con_constante_igual_al_individuo_sin_ella(datos):
    X, y, folds = datos
    fit = Fitness(X, y, folds, LinearRegression, _tree())
    sin = _ind([1, 1, 0, 0, 0, 0], [CONSTANTE], [0])
    con = _ind([1, 1, 0, 0, 0, 0], [CONSTANTE], [1])
    assert fit(con) == fit(sin)  # misma huella: la segunda llamada es acierto de caché
    assert fit.evaluations == 1


def test_fitness_sin_columnas_efectivas_pide_reparar(datos):
    X, y, folds = datos
    ind = _ind([0] * N, [CONSTANTE], [1])
    with pytest.raises(ValueError, match="repar"):
        Fitness(X, y, folds, LinearRegression, _tree())(ind)
