"""Tests del hito 8: piloto D20 (rejilla K × profundidad, solo train)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from evofeat import data

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_pilot", RAIZ / "scripts" / "run_pilot.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

PEQUENO = dict(pop_size=12, generations=2, hof_size=3, repeats=1)


def test_rejilla_y_semillas_del_diseno():
    assert pilot.REJILLA_K == (3, 5, 7) and pilot.REJILLA_PROFUNDIDAD == (2, 3, 4)
    assert pilot.SEMILLAS == (100, 101, 102)


def test_el_piloto_solo_usa_train(tmp_path, monkeypatch):
    """Si algo intentara leer validación, el piloto fallaría."""
    def prohibido(*a, **k):
        raise AssertionError("el piloto no puede leer validación")

    monkeypatch.setattr(data, "load_validation", prohibido)
    res = pilot.run_pilot("diabetes", tmp_path, jobs=1, **PEQUENO)
    assert len(res["rows"]) == 9
    texto = (RAIZ / "scripts" / "run_pilot.py").read_text()
    # el script no la importa ni la nombra más que en esta documentación
    assert not re.search(r"load_validation\s*\(", texto)


def test_piloto_completo_con_presupuesto_pequeno_y_reanudable(tmp_path):
    res = pilot.run_pilot("diabetes", tmp_path, jobs=1, **PEQUENO)
    archivos = sorted((tmp_path / "diabetes").glob("K*_seed*.json"))
    assert len(archivos) == 27 and res["executed_now"] == 27
    assert (tmp_path / "diabetes" / "summary.json").exists()
    r = json.loads(archivos[0].read_text())
    for campo in ("K", "max_depth", "seed", "cv_mae_best", "cv_fold_maes", "epsilon_run", "complexity",
                  "n_columns", "formulas", "seconds_fit", "distinct_evaluations", "budget"):
        assert campo in r
    assert len(r["cv_fold_maes"]) == 5 and r["seed"] in (100, 101, 102)
    assert r["epsilon_run"] == pytest.approx(np.std(r["cv_fold_maes"], ddof=1) / np.sqrt(5))
    assert r["individuals_generated"] == 12 + 2 * 10
    # relanzar no repite nada
    res2 = pilot.run_pilot("diabetes", tmp_path, jobs=1, **PEQUENO)
    assert res2["executed_now"] == 0 and res2["rows"] == res["rows"]
    assert not list(tmp_path.rglob("*.tmp"))
    # K y profundidad llegan realmente al transformador
    assert {(f["K"], f["max_depth"]) for f in res["rows"]} == {(k, d) for k in (3, 5, 7) for d in (2, 3, 4)}


def test_las_semillas_del_piloto_son_distintas_de_las_finales():
    assert not set(pilot.SEMILLAS) & set(range(30))


def _reg(K, d, cv, eps=0.01, seed=100):
    return {"K": K, "max_depth": d, "seed": seed, "cv_mae_best": cv, "epsilon_run": eps,
            "complexity": 10 * K, "n_columns": 4, "seconds_fit": 1.0, "distinct_evaluations": 100}


def test_seleccion_gana_el_mejor_mae_si_no_hay_empates():
    regs = [_reg(K, d, 1.0 + 0.1 * (K + d)) for K in (3, 5, 7) for d in (2, 3, 4)]
    regs.append(_reg(7, 4, 0.5))  # un valor excepcionalmente bueno
    res = pilot.resumir(regs)
    assert res["best_raw"] == {"K": 7, "max_depth": 4}
    assert res["proposed"] == {"K": 7, "max_depth": 4}


def test_seleccion_con_empate_elige_menor_k_y_profundidad():
    regs = [_reg(K, d, 1.0) for K in (3, 5, 7) for d in (2, 3, 4)]
    for r in regs:
        if (r["K"], r["max_depth"]) == (7, 4):
            r["cv_mae_best"] = 0.995  # el mejor, pero dentro de ε = 0.01
    res = pilot.resumir(regs)
    assert res["best_raw"] == {"K": 7, "max_depth": 4}
    assert res["proposed"] == {"K": 3, "max_depth": 2}
    assert len(res["within_epsilon"]) == 9


def test_seleccion_el_umbral_es_estricto_y_epsilon_es_la_mediana():
    regs = [_reg(3, 2, 1.0, eps=0.02), _reg(3, 3, 1.0, eps=0.01), _reg(5, 2, 0.99, eps=0.03)]
    res = pilot.resumir(regs)
    assert res["epsilon_proposed"] == pytest.approx(0.02)  # mediana de {0.02, 0.01, 0.03}
    # (3,2) y (3,3) están a 0.01 del mejor, que es < 0.02 -> dentro; se elige K = 3, profundidad 2
    assert res["proposed"] == {"K": 3, "max_depth": 2}


def test_la_media_se_hace_por_combinacion():
    regs = [_reg(3, 2, 1.0, seed=100), _reg(3, 2, 2.0, seed=101), _reg(3, 2, 3.0, seed=102)]
    f = pilot.resumir(regs)["rows"][0]
    assert f["cv_mae_mean"] == pytest.approx(2.0) and f["cv_mae_sd"] == pytest.approx(1.0) and f["runs"] == 3


def test_tabla_contiene_la_propuesta():
    regs = [_reg(K, d, 1.0 + 0.1 * (K + d)) for K in (3, 5, 7) for d in (2, 3, 4)]
    texto = pilot.tabla(pilot.resumir(regs), "diabetes")
    assert "| 3 | 2 |" in texto and "Propuesta" in texto and "ε propuesto" in texto


# --- baseline, regla común, comprobación de ε (hito 8b) -----------------------------------------

def test_baseline_cv_usa_ols_con_todos_los_originales_y_los_folds_repetidos():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = data.load_train("diabetes")
    b = pilot.baseline_cv("diabetes", seeds=(100,))
    maes = []
    for r in range(3):  # 3 repeticiones, random_state = semilla + 10 000 + r (D7)
        for entrenar, probar in KFold(5, shuffle=True, random_state=100 + 10_000 + r).split(tr.X):
            m = make_pipeline(StandardScaler(), LinearRegression()).fit(tr.X[entrenar], tr.y[entrenar])
            maes.append(np.mean(np.abs(tr.y[probar] - m.predict(tr.X[probar]))))
    assert b["mean"] == pytest.approx(np.mean(maes), rel=1e-10) and len(b["per_seed"]) == 1


def test_el_resumen_incluye_la_mejora_frente_al_baseline():
    regs = [_reg(K, d, 1.0 + 0.1 * (K + d)) for K in (3, 5, 7) for d in (2, 3, 4)]
    res = pilot.resumir(regs, {"per_seed": [2.0], "mean": 2.0})
    assert res["baseline_cv"]["mean"] == 2.0
    assert res["rows"][0]["vs_baseline_relative"] == pytest.approx((2.0 - res["rows"][0]["cv_mae_mean"]) / 2.0)
    assert "vs baseline" in pilot.tabla(res, "x") and "Baseline" in pilot.tabla(res, "x")


def test_regla_comun_interseccion_de_los_datasets():
    a = {"within_epsilon": [{"K": K, "max_depth": d} for K in (3, 5, 7) for d in (2, 3, 4)]}  # todo empata
    b = {"within_epsilon": [{"K": 3, "max_depth": 3}, {"K": 7, "max_depth": 3}]}
    r = pilot.proponer_comun({"diabetes": a, "california": b})
    assert r["proposed"] == {"K": 3, "max_depth": 3}
    assert r["valid"] == [{"K": 3, "max_depth": 3}, {"K": 7, "max_depth": 3}]
    vacia = pilot.proponer_comun({"a": {"within_epsilon": [{"K": 3, "max_depth": 2}]},
                                  "b": {"within_epsilon": [{"K": 7, "max_depth": 4}]}})
    assert vacia["proposed"] is None


def test_comprobacion_de_epsilon_con_presupuesto_pequeno(tmp_path):
    res = pilot.comprobar_epsilon("diabetes", 1.26, tmp_path, jobs=1, seeds=(100, 101), **PEQUENO)
    assert (tmp_path / "diabetes" / "eps_check.json").exists()
    assert res["K"] == 3 and res["max_depth"] == 3 and res["epsilon"] == 1.26
    # ε = 0 (pilot) y ε final en archivos distintos, y ε llega al transformador
    cero = json.loads((tmp_path / "diabetes" / "K3_d3_seed100.json").read_text())
    final = json.loads((tmp_path / "diabetes" / "K3_d3_eps_seed100.json").read_text())
    assert cero["epsilon"] == 0.0 and final["epsilon"] == 1.26
    assert res["difference_cv_mae"] == pytest.approx(
        res["epsilon_final"]["cv_mae_mean"] - res["epsilon_zero"]["cv_mae_mean"])
    assert res["stop"] == (res["difference_cv_mae"] > 1.26)
    assert "CRITERIO" in pilot.tabla_epsilon(res)
    # relanzar no repite
    again = pilot.comprobar_epsilon("diabetes", 1.26, tmp_path, jobs=1, seeds=(100, 101), **PEQUENO)
    assert again["epsilon_final"] == res["epsilon_final"]


def test_nucleos_fisicos_razonable():
    assert 1 <= pilot.nucleos_fisicos() <= 256
