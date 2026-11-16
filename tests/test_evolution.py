"""Tests del hito 6: bucle evolutivo (§8, D7, D13, D17, D19, D24, D25)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from evofeat.evolution import Evolution, EvolutionParams
from evofeat.fitness import Fitness
from evofeat.individual import Individual, build_matrix, n_columns
from evofeat.primitives import FUNCTION_NAMES, PRIMITIVES
from evofeat.repr_gep import GepRepresentation
from evofeat.repr_tree import TreeRepresentation

N = 5


@pytest.fixture(scope="module")
def sintetico():
    """5 atributos, 500 filas, y = x0·x1 + ruido pequeño (criterio del hito 6)."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(500, N))
    y = X[:, 0] * X[:, 1] + 0.05 * rng.normal(size=500)
    return X, y


def _tree():
    return TreeRepresentation(N, 3, FUNCTION_NAMES)


def _gep():
    return GepRepresentation(N, 7, FUNCTION_NAMES)


def _correr(X, y, rep, configuracion="C", seed=0, max_cached=1000, **kw):
    params = EvolutionParams.for_configuration(configuracion, **{"pop_size": 30, "generations": 8, **kw})
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    fit = Fitness(X, y, folds, LinearRegression, rep, max_cached_values=max_cached)
    res = Evolution(params, rep, fit, N, np.random.default_rng(seed)).run()
    return res, fit


# --- criterios de aceptación ------------------------------------------------------------

def test_problema_sintetico_C_encuentra_la_interaccion_en_8_de_10_semillas(sintetico):
    """y = x0·x1: la mejor solución reproduce x0·x1 (R² ≥ 0,99 sobre las columnas de Xt′)."""
    X, y = sintetico
    objetivo = X[:, 0] * X[:, 1]
    aciertos = 0
    for seed in range(10):
        res, fit = _correr(X, y, _tree(), seed=seed, pop_size=50, generations=20)
        M = np.column_stack([build_matrix(res.best, fit.evaluator), np.ones(len(y))])
        coef = np.linalg.lstsq(M, objetivo, rcond=None)[0]
        r2 = 1 - np.var(objetivo - M @ coef) / np.var(objetivo)
        aciertos += r2 >= 0.99
    assert aciertos >= 8


@pytest.mark.parametrize("rep_fn", [_tree, _gep])
def test_el_mejor_fitness_nunca_empeora(sintetico, rep_fn):
    X, y = sintetico
    res, _ = _correr(X, y, rep_fn(), generations=12)
    mejores = [h["best_fitness"] for h in res.history]
    assert all(b <= a + 1e-15 for a, b in zip(mejores, mejores[1:]))
    assert mejores[-1] == res.best_fitness == min(res.fitnesses)


@pytest.mark.parametrize("rep_fn", [_tree, _gep])
def test_misma_semilla_mismo_resultado(sintetico, rep_fn):
    X, y = sintetico
    a, _ = _correr(X, y, rep_fn(), seed=3)
    b, _ = _correr(X, y, rep_fn(), seed=3)
    c, _ = _correr(X, y, rep_fn(), seed=4)
    assert a.history == b.history
    assert a.best_fitness == b.best_fitness
    assert [i.genomes for i in a.population] == [i.genomes for i in b.population]
    assert a.history != c.history


def test_el_tamano_de_la_cache_lru_solo_afecta_al_tiempo(sintetico):
    """Misma semilla con LRU de tamaño 1 y de 1000: resultados exactamente iguales."""
    X, y = sintetico
    for rep_fn in (_tree, _gep):
        a, fa = _correr(X, y, rep_fn(), seed=5, max_cached=1)
        b, fb = _correr(X, y, rep_fn(), seed=5, max_cached=1000)
        assert a.history == b.history
        assert a.best_fitness == b.best_fitness
        assert a.fitnesses == b.fitnesses
        assert [i.genomes for i in a.population] == [i.genomes for i in b.population]
        assert (fa.evaluations, fa.cache_hits) == (fb.evaluations, fb.cache_hits)


# --- estructura de la población y límites -------------------------------------------------

def test_poblacion_inicial_con_individuos_semilla(sintetico):
    X, y = sintetico
    params = EvolutionParams.for_configuration("C", pop_size=30, generations=0)
    rep = _tree()
    fit = Fitness(X, y, list(KFold(5, shuffle=True, random_state=0).split(X)), LinearRegression, rep)
    ev = Evolution(params, rep, fit, N, np.random.default_rng(0))
    pob = ev._inicial()
    assert len(pob) == 30
    semillas = pob[:N]
    for j, ind in enumerate(semillas):
        assert ind.mask.sum() == N - 1 and not ind.mask[j]  # un original eliminado cada una
        assert not ind.switches.any()  # ningún árbol activo
    # las n semillas cubren todas las formas de quitar un solo original
    assert len({tuple(i.mask) for i in semillas}) == N


@pytest.mark.parametrize("rep_fn", [_tree, _gep])
def test_configuracion_C_respeta_el_limite_de_columnas(sintetico, rep_fn):
    X, y = sintetico
    res, fit = _correr(X, y, rep_fn(), "C", generations=6)
    for ind in res.population + [i for _, i in res.hall_of_fame]:
        assert 1 <= n_columns(ind, fit.evaluator) <= N - 1


def test_configuracion_A_solo_selecciona(sintetico):
    X, y = sintetico
    res, fit = _correr(X, y, _tree(), "A", generations=6)
    for ind in res.population:
        assert not ind.switches.any()
        assert 1 <= ind.mask.sum() <= N - 1


def test_configuracion_B_mascara_fija_y_sin_limite(sintetico):
    X, y = sintetico
    res, fit = _correr(X, y, _tree(), "B", generations=6)
    assert all(ind.mask.all() for ind in res.population)
    # D24 no aplica a B: puede tener n o más columnas
    assert max(n_columns(ind, fit.evaluator) for ind in res.population) >= N


def test_configuracion_desconocida():
    with pytest.raises(ValueError):
        EvolutionParams.for_configuration("Z")


@pytest.mark.parametrize("rep_fn", [_tree, _gep])
def test_genes_siguen_siendo_validos_tras_evolucionar(sintetico, rep_fn):
    X, y = sintetico
    rep = rep_fn()
    res, _ = _correr(X, y, rep, generations=8)
    for ind in res.population:
        for g in ind.genomes:
            f = rep.to_formula(g)
            assert f[0] in PRIMITIVES and len(f) <= 15
            if rep.kind == "gep":
                assert all(s not in PRIMITIVES for s in g[rep.head_len:])


# --- elitismo, hall of fame, historial -----------------------------------------------------

def test_elitismo_conserva_a_los_dos_mejores(sintetico):
    X, y = sintetico
    res, _ = _correr(X, y, _tree(), generations=1)
    anterior, _ = _correr(X, y, _tree(), generations=0)
    dos_mejores = sorted(anterior.fitnesses)[:2]
    for v in dos_mejores:
        assert v in res.fitnesses


def test_hall_of_fame(sintetico):
    X, y = sintetico
    res, fit = _correr(X, y, _tree(), generations=10)
    valores = [v for v, _ in res.hall_of_fame]
    assert len(valores) == 10 and valores == sorted(valores)
    assert valores[0] == res.best_fitness  # el mejor visto es el mejor final (elitismo)
    claves = {tuple(i.mask) + tuple(i.switches) for _, i in res.hall_of_fame}
    assert len(res.hall_of_fame) == 10 and len(claves) >= 1


def test_historial_y_presupuesto(sintetico):
    X, y = sintetico
    res, fit = _correr(X, y, _tree(), generations=7)
    assert [h["generation"] for h in res.history] == list(range(8))
    for h in res.history:
        assert set(h) == {"generation", "best_fitness", "median_fitness", "best_complexity",
                          "cache_hits", "evaluations", "constants_dropped"}
        assert h["best_fitness"] <= h["median_fitness"]
        # evaluaciones reales + aciertos de caché = individuos evaluados en la generación
        esperado = 30 if h["generation"] == 0 else 30 - 2
        assert h["evaluations"] + h["cache_hits"] == esperado
    assert fit.evaluations == sum(h["evaluations"] for h in res.history)


def test_el_historial_cuenta_constantes_descartadas(sintetico):
    X, y = sintetico
    res, fit = _correr(X, y, _tree(), generations=6)
    assert all(h["constants_dropped"] >= 0 for h in res.history)


# --- selección con parsimonia lexicográfica ----------------------------------------------------

def _evo_con_epsilon(sintetico, eps):
    X, y = sintetico
    rep = _tree()
    fit = Fitness(X, y, list(KFold(5, shuffle=True, random_state=0).split(X)), LinearRegression, rep)
    return Evolution(EvolutionParams(epsilon=eps), rep, fit, N, np.random.default_rng(0))


def test_duelo_sin_epsilon_gana_el_mejor_fitness(sintetico):
    evo = _evo_con_epsilon(sintetico, 0.0)
    fit, comp = [1.0, 1.0001], [50, 5]
    assert evo._gana(0, 1, fit, comp) == 0  # sin tolerancia, solo cuenta el fitness


def test_duelo_con_epsilon_gana_el_mas_simple_si_estan_empatados(sintetico):
    evo = _evo_con_epsilon(sintetico, 0.01)
    fit, comp = [1.0, 1.005], [50, 5]
    assert evo._gana(0, 1, fit, comp) == 1  # diferencia < ε: gana el de menor complejidad
    fit2 = [1.0, 1.5]
    assert evo._gana(0, 1, fit2, comp) == 0  # diferencia ≥ ε: gana el mejor fitness
    assert evo._gana(0, 1, [1.0, 1.005], [7, 7]) == 0  # misma complejidad: el mejor fitness


def test_torneo_devuelve_indices_validos_y_prefiere_mejores(sintetico):
    evo = _evo_con_epsilon(sintetico, 0.0)
    fit = [float(i) for i in range(100)]  # el 0 es el mejor
    comp = [10] * 100
    elegidos = [evo._torneo(fit, comp) for _ in range(2000)]
    assert all(0 <= e < 100 for e in elegidos)
    assert np.mean(elegidos) < 30  # sesgo claro hacia los mejores (media uniforme sería ~50)


# --- operadores a nivel de individuo ------------------------------------------------------------

def test_gen_e_interruptor_viajan_juntos(sintetico):
    rep = _tree()
    ga = tuple(("x%d" % i,) for i in range(5))
    a = Individual(np.ones(N, bool), tuple(("sqrt", "x%d" % i) for i in range(5)),
                   np.array([1, 0, 1, 0, 1], dtype=bool))
    b = Individual(np.zeros(N, bool), tuple(("log", "x%d" % i) for i in range(5)),
                   np.array([0, 1, 0, 1, 0], dtype=bool))
    h1, h2 = Evolution._intercambiar_gen(a, b, 2)
    assert h1.genomes[2] == b.genomes[2] and h2.genomes[2] == a.genomes[2]
    assert h1.switches[2] == b.switches[2] and h2.switches[2] == a.switches[2]
    for k in (0, 1, 3, 4):  # el resto no cambia
        assert h1.genomes[k] == a.genomes[k] and h1.switches[k] == a.switches[k]
    assert a.switches.tolist() == [True, False, True, False, True]  # los originales no se modifican


def test_mutacion_de_bits_tasa_uno_sobre_n_mas_k(sintetico):
    evo = _evo_con_epsilon(sintetico, 0.0)
    rng = np.random.default_rng(9)
    base = Individual(np.zeros(N, bool), tuple(evo.rep.random_genome(rng) for _ in range(5)), np.zeros(5, bool))
    cambios = 0
    reps = 4000
    for _ in range(reps):
        m = evo._mutacion(base)
        cambios += int(m.mask.sum() + m.switches.sum())
    # en media 1 bit por individuo (1/(n+K) por bit sobre n+K bits)
    assert 0.9 < cambios / reps < 1.1


def test_cruce_uniforme_de_mascara_conserva_los_bits_entre_los_hijos(sintetico):
    evo = _evo_con_epsilon(sintetico, 0.0)
    rng = np.random.default_rng(10)
    a = Individual(np.array([1, 1, 1, 1, 0], bool), tuple(evo.rep.random_genome(rng) for _ in range(5)), np.zeros(5, bool))
    b = Individual(np.array([0, 0, 1, 0, 1], bool), a.genomes, np.zeros(5, bool))
    for _ in range(200):
        h1, h2 = evo._cruce(a, b)
        assert all({x, y} == {p, q} for x, y, p, q in zip(h1.mask, h2.mask, a.mask, b.mask))
