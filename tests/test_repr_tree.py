"""Tests del hito 3: representación en árbol (§8, D13, RF-06)."""

from __future__ import annotations

import numpy as np
import pytest

from evofeat.formula import depth, is_terminal, is_valid
from evofeat.primitives import FUNCTION_NAMES, PRIMITIVES
from evofeat.repr_tree import TreeRepresentation, _alturas, _fin_subarbol, _profundidades_nodos

N = 8


def _rep(max_depth: int = 3) -> TreeRepresentation:
    return TreeRepresentation(N, max_depth, FUNCTION_NAMES)


def _terminales_validos(g) -> bool:
    return all(not is_terminal(s) or int(s[1:]) < N for s in g)


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_grow_respeta_profundidad(max_depth):
    rep, rng = _rep(max_depth), np.random.default_rng(1)
    for _ in range(500):
        g = rep.grow(rng)
        assert is_valid(g) and depth(g) <= max_depth and _terminales_validos(g)


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_full_hojas_a_profundidad_maxima(max_depth):
    rep, rng = _rep(max_depth), np.random.default_rng(2)
    for _ in range(500):
        g = rep.full(rng)
        assert is_valid(g) and depth(g) == max_depth
        profs = _profundidades_nodos(g)
        assert all(p == max_depth for s, p in zip(g, profs) if is_terminal(s))


def test_grow_produce_variedad_de_profundidades():
    rep, rng = _rep(3), np.random.default_rng(3)
    assert {depth(rep.grow(rng)) for _ in range(500)} >= {1, 2, 3}


def test_ramped_half_and_half_cubre_profundidades_1_a_max():
    rep, rng = _rep(3), np.random.default_rng(4)
    gs = [rep.random_genome(rng) for _ in range(2000)]
    assert all(is_valid(g) and 1 <= depth(g) <= 3 for g in gs)
    assert {depth(g) for g in gs} == {1, 2, 3}


def test_helpers_de_subarbol():
    f = ("add", "sqrt", "x0", "mul", "x1", "x2")
    assert _profundidades_nodos(f) == [0, 1, 2, 1, 2, 2]
    assert _fin_subarbol(f, 0) == 6
    assert _fin_subarbol(f, 1) == 3
    assert _fin_subarbol(f, 3) == 6
    assert _fin_subarbol(f, 5) == 6


def test_estres_10000_operaciones_nunca_superan_profundidad():
    """10 000 cruces y mutaciones aleatorios: preorden válido, profundidad ≤ 3, raíz función."""
    rep, rng = _rep(3), np.random.default_rng(5)
    # mitad Full (profundidad máxima, el caso más exigente para el cruce) y mitad ramped
    pob = [rep.full(rng) for _ in range(25)] + [rep.random_genome(rng) for _ in range(25)]
    operaciones = 0
    while operaciones < 10_000:
        if rng.random() < 0.5:
            a, b = pob[int(rng.integers(50))], pob[int(rng.integers(50))]
            h1, h2 = rep.crossover(a, b, rng)
            hijos = [h1, h2]
            operaciones += 2
        else:
            hijos = [rep.mutate(pob[int(rng.integers(50))], rng)]
            operaciones += 1
        for h in hijos:
            assert is_valid(h), h
            assert depth(h) <= 3, h
            assert h[0] in PRIMITIVES, h  # la raíz es siempre una función
            assert _terminales_validos(h)
            pob[int(rng.integers(50))] = h  # realimenta la población


def test_raiz_siempre_funcion_en_inicializacion():
    rep, rng = _rep(3), np.random.default_rng(12)
    assert all(rep.random_genome(rng)[0] in PRIMITIVES for _ in range(1000))


def test_tasa_de_cruce_efectiva_es_la_nominal():
    """El cruce nunca necesita la red de seguridad: casi siempre cambia a algún hijo."""
    rep, rng = _rep(3), np.random.default_rng(13)
    cambia = 0
    total = 2000
    for _ in range(total):
        a, b = rep.full(rng), rep.full(rng)
        h1, h2 = rep.crossover(a, b, rng)
        cambia += (h1 != a) or (h2 != b)
    assert cambia / total > 0.9


def test_cruce_intercambia_subarboles_y_conserva_simbolos():
    rep, rng = _rep(3), np.random.default_rng(6)
    a, b = rep.full(rng), rep.full(rng)
    cambios = 0
    for _ in range(200):
        h1, h2 = rep.crossover(a, b, rng)
        cambios += (h1 != a) + (h2 != b)
        # cruce de subárbol (sin rechazo) conserva el multiconjunto total de símbolos
        if h1 != a and h2 != b:
            assert sorted(h1 + h2) == sorted(a + b)
    assert cambios > 0


def test_cruce_con_limite_estricto_siempre_posible():
    """Con max_depth = 1 casi todo cruce ingenuo se pasaría; aquí siempre hay un punto válido."""
    rep = TreeRepresentation(2, 1, ("add", "sub"))
    rng = np.random.default_rng(7)
    a, b = ("add", "x0", "x1"), ("sub", "x1", "x1")
    for _ in range(300):
        h1, h2 = rep.crossover(a, b, rng)
        for h in (h1, h2):
            assert is_valid(h) and depth(h) == 1 and h[0] in PRIMITIVES


def test_alturas():
    f = ("add", "sqrt", "x0", "mul", "x1", "x2")
    assert _alturas(f) == [2, 1, 0, 1, 0, 0]
    assert _alturas(("x0",)) == [0]


def test_mutacion_puntual_cambia_un_nodo_con_igual_aridad():
    rep, rng = _rep(3), np.random.default_rng(8)
    g = ("div", "log", "mul", "x0", "x1", "sqrt", "x2")
    cambiados = 0
    for _ in range(300):
        h = rep._mutacion_puntual(g, rng)
        diffs = [i for i in range(len(g)) if g[i] != h[i]]
        assert len(h) == len(g) and len(diffs) <= 1
        if diffs:
            i = diffs[0]
            cambiados += 1
            if is_terminal(g[i]):
                assert is_terminal(h[i])
            else:
                assert PRIMITIVES[g[i]][0] == PRIMITIVES[h[i]][0]
        assert is_valid(h)
    assert cambiados > 0


def test_mutacion_puntual_sin_alternativas_no_cambia():
    rep = TreeRepresentation(1, 2, ("add",))
    g = ("add", "x0", "x0")
    rng = np.random.default_rng(9)
    assert all(rep._mutacion_puntual(g, rng) == g for _ in range(20))


def test_mutacion_de_subarbol_en_la_raiz_deja_funcion():
    rep, rng = _rep(3), np.random.default_rng(14)
    g = ("add", "x0", "x1")
    # forzamos el punto raíz: con un árbol de 3 nodos y muchas repeticiones sale casi siempre
    assert all(rep._mutacion_subarbol(g, rng)[0] in PRIMITIVES for _ in range(500))


def test_mutacion_de_subarbol_explora_formas_nuevas():
    rep, rng = _rep(3), np.random.default_rng(10)
    g = rep.full(rng)
    formas = {len(rep._mutacion_subarbol(g, rng)) for _ in range(300)}
    assert len(formas) > 1


def test_reproducibilidad_misma_semilla_mismo_resultado():
    def corrida(semilla):
        rep, rng = _rep(3), np.random.default_rng(semilla)
        pob = [rep.random_genome(rng) for _ in range(10)]
        out = []
        for _ in range(100):
            h1, h2 = rep.crossover(pob[int(rng.integers(10))], pob[int(rng.integers(10))], rng)
            out += [h1, h2, rep.mutate(h1, rng)]
        return out

    assert corrida(11) == corrida(11)
    assert corrida(11) != corrida(12)


def test_to_formula_es_identidad_y_validacion_de_parametros():
    rep = _rep()
    g = rep.random_genome(np.random.default_rng(0))
    assert rep.to_formula(g) == g
    for kwargs in [(0, 3, FUNCTION_NAMES), (4, 0, FUNCTION_NAMES), (4, 3, ("foo",)), (4, 3, ())]:
        with pytest.raises(ValueError):
            TreeRepresentation(*kwargs)
