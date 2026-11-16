"""Tests del hito 2: primitivas protegidas y fórmulas (D14, D15, RF-04, RNF-04)."""

from __future__ import annotations

import numpy as np
import pytest

from evofeat import primitives as P
from evofeat.formula import depth, evaluate, fingerprint, is_valid, size, to_infix

VALORES_EXTREMOS = np.array([0.0, 1e-12, -1e-12, 1e12, -1e12, 1.0, -1.0, 2.5, -3.7])


def _todas_las_parejas():
    a, b = np.meshgrid(VALORES_EXTREMOS, VALORES_EXTREMOS)
    return a.ravel(), b.ravel()


# --- primitivas protegidas -------------------------------------------------

@pytest.mark.parametrize("nombre", ["add", "sub", "mul", "div"])
def test_binarias_finitas_con_valores_extremos(nombre):
    a, b = _todas_las_parejas()
    r = P.PRIMITIVES[nombre][1](a, b)
    assert np.isfinite(r).all()


@pytest.mark.parametrize("nombre", ["sqrt", "log"])
def test_unarias_finitas_con_valores_extremos(nombre):
    r = P.PRIMITIVES[nombre][1](VALORES_EXTREMOS)
    assert np.isfinite(r).all()


def test_division_protegida():
    a = np.array([5.0, 5.0, 5.0, 5.0, -4.0])
    b = np.array([0.0, 1e-12, -1e-12, 5e-7, 2.0])
    np.testing.assert_array_equal(P.div(a, b), [1.0, 1.0, 1.0, 1.0, -2.0])
    # justo en el umbral (|b| = 1e-6) ya no se protege: la condición es estricta (<)
    np.testing.assert_allclose(P.div(np.array([1.0]), np.array([1e-6])), [1e6])
    np.testing.assert_allclose(P.div(np.array([1.0]), np.array([-1e-6])), [-1e6])


def test_sqrt_y_log_con_negativos():
    x = np.array([-9.0, 0.0, 4.0])
    np.testing.assert_allclose(P.sqrt(x), [3.0, 0.0, 2.0])  # √|x|
    np.testing.assert_allclose(P.log(x), [np.log(10.0), 0.0, np.log(5.0)])  # log(1+|x|)


def test_desbordamiento_acotado_sin_inf_ni_nan():
    g = np.array([1e300, -1e300])
    for nombre in ("add", "sub", "mul"):
        r = P.PRIMITIVES[nombre][1](g, g)
        assert np.isfinite(r).all() and (np.abs(r) <= P.LIMITE).all()
    # encadenar desbordamientos sigue siendo finito
    f = ("mul", "mul", "x0", "x0", "mul", "x0", "x0")
    assert np.isfinite(evaluate(f, g.reshape(-1, 1))).all()


def test_aridades():
    assert [P.arity(n) for n in P.FUNCTION_NAMES] == [2, 2, 2, 2, 1, 1]
    assert set(P.FUNCTION_NAMES) == {"add", "sub", "mul", "div", "sqrt", "log"}


# --- evaluate: tres fórmulas contra cálculo a mano ------------------------

X = np.array([[1.0, 2.0, 4.0],
              [3.0, -1.0, 0.0],
              [-2.0, 5.0, 2.0]])


def test_evaluate_terminal_simple():
    np.testing.assert_array_equal(evaluate(("x1",), X), [2.0, -1.0, 5.0])


def test_evaluate_a_mano_1_resta_no_conmutativa():
    # (x0 - x1) * x2 -> orden de operandos correcto
    f = ("mul", "sub", "x0", "x1", "x2")
    esperado = (X[:, 0] - X[:, 1]) * X[:, 2]
    np.testing.assert_allclose(evaluate(f, X), esperado)
    np.testing.assert_allclose(evaluate(f, X), [-4.0, 0.0, -14.0])


def test_evaluate_a_mano_2_division_protegida_en_fila():
    # x0 / x2: la fila 1 tiene x2 = 0 -> protegida, vale 1
    f = ("div", "x0", "x2")
    np.testing.assert_allclose(evaluate(f, X), [0.25, 1.0, -1.0])


def test_evaluate_a_mano_3_anidada_con_unarias():
    # log(1 + |x0 * x1|) + sqrt(|x1|)
    f = ("add", "log", "mul", "x0", "x1", "sqrt", "x1")
    esperado = np.log1p(np.abs(X[:, 0] * X[:, 1])) + np.sqrt(np.abs(X[:, 1]))
    np.testing.assert_allclose(evaluate(f, X), esperado)
    np.testing.assert_allclose(evaluate(f, X)[1], np.log(4.0) + 1.0)


def test_evaluate_no_modifica_X_y_es_vector():
    copia = X.copy()
    r = evaluate(("x0",), X)
    r += 100.0  # modificar el resultado no debe tocar X
    np.testing.assert_array_equal(X, copia)
    assert evaluate(("add", "x0", "x1"), X).shape == (3,)


def test_evaluate_finito_con_datos_aleatorios_extremos():
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(200, 3)) * 10.0 ** rng.integers(-12, 13, size=(200, 3))
    Xr[0] = 0.0
    for f in [("div", "x0", "sub", "x1", "x1"), ("log", "div", "x0", "x1"),
              ("mul", "mul", "x0", "x1", "mul", "x1", "x2")]:
        assert np.isfinite(evaluate(f, Xr)).all()


# --- depth, size, to_infix ----------------------------------------------------

def test_depth_y_size():
    assert (depth(("x0",)), size(("x0",))) == (0, 1)
    assert (depth(("sqrt", "x0")), size(("sqrt", "x0"))) == (1, 2)
    f = ("div", "log", "mul", "x0", "x1", "x2")
    assert (depth(f), size(f)) == (3, 6)
    # árbol completo de profundidad 3: 15 nodos
    completo = ("add",
                "add", "add", "x0", "x1", "add", "x2", "x3",
                "add", "add", "x4", "x5", "add", "x6", "x7")
    assert (depth(completo), size(completo)) == (3, 15)
    # rama izquierda profunda y derecha corta
    assert depth(("add", "sqrt", "sqrt", "x0", "x1")) == 3  # add > sqrt > sqrt > x0


def test_to_infix():
    nombres = ["MedInc", "HouseAge", "Population"]
    assert to_infix(("x1",), nombres) == "HouseAge"
    assert to_infix(("sub", "x0", "x1"), nombres) == "(MedInc - HouseAge)"
    f = ("div", "log", "mul", "x0", "x1", "x2")
    assert to_infix(f, nombres) == "(log(1 + |(MedInc * HouseAge)|) / Population)"
    assert to_infix(("add", "sqrt", "x0", "div", "x1", "x2"), nombres) == \
        "(sqrt(|MedInc|) + (HouseAge / Population))"


# --- validez de formulas y huella ---------------------------------------------

@pytest.mark.parametrize("f", [
    (), ("add",), ("add", "x0"), ("x0", "x1"), ("sqrt", "x0", "x1"),
    ("foo", "x0"), ("add", "x0", "x"), ("x",),
])
def test_formulas_mal_formadas(f):
    assert not is_valid(f)
    for fun in (depth, size):
        with pytest.raises(ValueError):
            fun(f)
    with pytest.raises(ValueError):
        evaluate(f, X)
    with pytest.raises(ValueError):
        to_infix(f, ["a", "b", "c"])


def test_formulas_bien_formadas():
    assert is_valid(("x0",)) and is_valid(("div", "x0", "log", "x2"))


def test_fingerprint_distingue_formulas():
    a, b = ("sub", "x0", "x1"), ("sub", "x1", "x0")
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) == fingerprint(("sub", "x0", "x1"))
