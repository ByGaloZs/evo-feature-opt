"""Tests del hito 4: representación GEP y decodificación Karva (§8, D25, RF-04, RF-06)."""

from __future__ import annotations

import numpy as np
import pytest

from evofeat.formula import evaluate, is_valid, size, to_infix
from evofeat.primitives import FUNCTION_NAMES, PRIMITIVES
from evofeat.repr_gep import GepRepresentation, decode_karva

N = 6
H = 7

SIMBOLOS = {"+": "add", "-": "sub", "*": "mul", "/": "div", "Q": "sqrt", "L": "log"}


def _gen(texto: str) -> tuple[str, ...]:
    """Convierte la notación de las diapositivas ('+/Q*c-abde') a símbolos del proyecto.
    Las letras a, b, c... son los terminales x0, x1, x2..."""
    return tuple(SIMBOLOS[c] if c in SIMBOLOS else f"x{ord(c) - ord('a')}" for c in texto)


NOMBRES = list("abcde")


def _rep() -> GepRepresentation:
    return GepRepresentation(N, H, FUNCTION_NAMES)


def _gen_valido(rep: GepRepresentation, g) -> bool:
    cabeza, cola = g[: rep.head_len], g[rep.head_len :]
    return (
        len(g) == rep.gene_len
        and g[0] in PRIMITIVES  # raíz función
        and all(s not in PRIMITIVES and s in rep.terminals for s in cola)  # cola: solo terminales
        and all(s in PRIMITIVES or s in rep.terminals for s in cabeza)
    )


# --- decodificación: ejemplos de las diapositivas -------------------------

def test_karva_ejemplo_1():
    f = decode_karva(_gen("+/Q*c-abde"))
    assert to_infix(f, NOMBRES) == "(((a * b) / c) + sqrt(|(d - e)|))"
    assert size(f) == 10


def test_karva_ejemplo_2():
    f = decode_karva(_gen("*-/Qb+b+aaab"))
    assert to_infix(f, NOMBRES) == "((sqrt(|(a + b)|) - b) * ((a + a) / b))"


def test_karva_ejemplo_3_solo_8_simbolos_expresados():
    gen = _gen("/aQ/b*ab/Qa*b*-ababaababbabbbba")
    f = decode_karva(gen)
    assert len(f) == 8
    assert to_infix(f, NOMBRES) == "(a / sqrt(|(b / (a * b))|))"
    # los símbolos no expresados no cambian la fórmula
    modificado = gen[:8] + tuple(reversed(gen[8:]))
    assert decode_karva(modificado) == f


def test_karva_evaluacion_coincide_con_calculo_a_mano():
    X = np.array([[2.0, 3.0, 4.0, 10.0, 1.0], [1.0, 5.0, 2.0, 5.0, 1.0]])
    f = decode_karva(_gen("+/Q*c-abde"))
    esperado = X[:, 0] * X[:, 1] / X[:, 2] + np.sqrt(np.abs(X[:, 3] - X[:, 4]))
    np.testing.assert_allclose(evaluate(f, X), esperado)


def test_karva_gen_incompleto_falla():
    with pytest.raises(ValueError):
        decode_karva(("add", "x0"))  # la raíz pide 2 hijos y solo hay 1 símbolo


# --- generación y decodificación aleatoria ------------------------------------

def test_longitudes_cabeza_y_cola():
    rep = _rep()
    assert (rep.head_len, rep.tail_len, rep.gene_len) == (7, 8, 15)  # t = h·(2−1)+1
    assert GepRepresentation(N, 3, FUNCTION_NAMES).gene_len == 3 + 4
    assert GepRepresentation(N, 15, FUNCTION_NAMES).gene_len == 15 + 16


def test_todo_cromosoma_aleatorio_decodifica():
    rep, rng = _rep(), np.random.default_rng(1)
    for _ in range(5000):
        g = rep.random_genome(rng)
        assert _gen_valido(rep, g)
        f = rep.to_formula(g)
        assert is_valid(f) and size(f) <= 15 and f[0] in PRIMITIVES


def test_distintas_formulas_expresadas():
    rep, rng = _rep(), np.random.default_rng(2)
    tamanos = {size(rep.to_formula(rep.random_genome(rng))) for _ in range(500)}
    assert len(tamanos) > 5  # no siempre el mismo tamaño


# --- operadores respetan cabeza/cola ------------------------------------------

def test_estres_mutacion_y_recombinacion_conservan_estructura():
    """Tras 10 000 operaciones la cola solo tiene terminales y la raíz es función."""
    rep, rng = _rep(), np.random.default_rng(3)
    pob = [rep.random_genome(rng) for _ in range(40)]
    operaciones = 0
    while operaciones < 10_000:
        if rng.random() < 0.5:
            hijos = list(rep.crossover(pob[int(rng.integers(40))], pob[int(rng.integers(40))], rng))
            operaciones += 2
        else:
            hijos = [rep.mutate(pob[int(rng.integers(40))], rng)]
            operaciones += 1
        for h in hijos:
            assert _gen_valido(rep, h), h
            f = rep.to_formula(h)
            assert is_valid(f) and size(f) <= 15
            pob[int(rng.integers(40))] = h


def test_mutacion_puntual_tasa_y_cola_solo_terminales():
    rep = GepRepresentation(N, H, FUNCTION_NAMES, p_point=1.0)  # muta todos los símbolos
    rng = np.random.default_rng(4)
    g = rep.random_genome(rng)
    for _ in range(200):
        h = rep.point_mutation(g, rng)
        assert _gen_valido(rep, h)
    # con la tasa de Ferreira cambia de media ~0,044·15 ≈ 0,66 símbolos (como mucho)
    rep2, cambios = _rep(), 0
    for _ in range(2000):
        h = rep2.point_mutation(g, rng)
        cambios += sum(x != y for x, y in zip(g, h))
    assert 0.3 < cambios / 2000 < 0.9


def test_is_inserta_en_cabeza_sin_tocar_cola_ni_raiz():
    rep, rng = _rep(), np.random.default_rng(5)
    for _ in range(1000):
        g = rep.random_genome(rng)
        h = rep.is_transposition(g, rng)
        assert h[rep.head_len :] == g[rep.head_len :] and h[0] == g[0]
        assert len(h) == len(g) and _gen_valido(rep, h)


def test_ris_pone_una_funcion_en_la_raiz():
    rep, rng = _rep(), np.random.default_rng(6)
    movidos = 0
    for _ in range(1000):
        g = rep.random_genome(rng)
        h = rep.ris_transposition(g, rng)
        assert h[rep.head_len :] == g[rep.head_len :] and _gen_valido(rep, h)
        movidos += h != g
    assert movidos > 0


def test_ris_sin_funcion_posible_no_cambia():
    rep = GepRepresentation(2, 3, ("add",))
    g = ("add", "x0", "x1", "x0", "x1", "x0", "x1")
    # cabeza ('add','x0','x1'): desde el punto 1 o 2 no hay función; desde 0 sí
    rng = np.random.default_rng(7)
    resultados = {rep.ris_transposition(g, rng) for _ in range(200)}
    assert g in resultados and all(_gen_valido(rep, r) for r in resultados)


def test_recombinacion_un_punto_y_dos_puntos():
    rep, rng = _rep(), np.random.default_rng(8)
    for _ in range(500):
        a, b = rep.random_genome(rng), rep.random_genome(rng)
        for op in (rep.one_point_recombination, rep.two_point_recombination):
            h1, h2 = op(a, b, rng)
            assert _gen_valido(rep, h1) and _gen_valido(rep, h2)
            # posiciones homólogas: en cada posición, {h1, h2} es {a, b}
            assert all({x, y} == {p, q} for x, y, p, q in zip(h1, h2, a, b))


def test_crossover_con_tasas_extremas():
    rng = np.random.default_rng(9)
    sin = GepRepresentation(N, H, FUNCTION_NAMES, p_one_point=0.0, p_two_point=0.0)
    a, b = sin.random_genome(rng), sin.random_genome(rng)
    assert sin.crossover(a, b, rng) == (a, b)
    con = GepRepresentation(N, H, FUNCTION_NAMES, p_one_point=1.0, p_two_point=1.0)
    assert any(con.crossover(a, b, rng) != (a, b) for _ in range(20))


def test_mutate_sin_operadores_es_identidad():
    rep = GepRepresentation(N, H, FUNCTION_NAMES, p_point=0.0, p_is=0.0, p_ris=0.0)
    rng = np.random.default_rng(10)
    g = rep.random_genome(rng)
    assert rep.mutate(g, rng) == g


def test_reproducibilidad_misma_semilla_mismo_resultado():
    def corrida(semilla):
        rep, rng = _rep(), np.random.default_rng(semilla)
        pob = [rep.random_genome(rng) for _ in range(10)]
        out = []
        for _ in range(100):
            h1, h2 = rep.crossover(pob[int(rng.integers(10))], pob[int(rng.integers(10))], rng)
            out += [h1, h2, rep.mutate(h1, rng)]
        return out

    assert corrida(11) == corrida(11)
    assert corrida(11) != corrida(12)


def test_validacion_de_parametros():
    for args in [(0, 7, FUNCTION_NAMES), (4, 1, FUNCTION_NAMES), (4, 7, ("foo",)), (4, 7, ())]:
        with pytest.raises(ValueError):
            GepRepresentation(*args)
