"""Tests del hito 10: análisis de resultados (D22, D25, §11)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze", RAIZ / "scripts" / "analyze.py")
an = importlib.util.module_from_spec(_spec)
sys.modules["analyze"] = an
_spec.loader.exec_module(an)

DATASETS = {"diabetes": (10, 40.0), "california": (8, 0.5)}  # (atributos, e)


# --- datos sintéticos con efectos conocidos --------------------------------------------------------

def _escribir_runs(dir_runs: Path, efectos: dict[str, float] | None = None, n: int = 30) -> None:
    """e′ = e · (1 − efecto) + ruido; efecto > 0 mejora. Genera los 5 experimentos × 2 datasets."""
    efectos = efectos or {"baseline": 0.0, "A": 0.0, "B": 0.10, "C_tree": 0.07, "C_gep": 0.07}
    rng = np.random.default_rng(0)
    for ds, (n_attr, e) in DATASETS.items():
        nombres = [c for c in __import__("evofeat.data", fromlist=["x"]).load_raw(ds).nombres]
        for clave, efecto in efectos.items():
            conf, rep = ("C", "tree") if clave == "C_tree" else ("C", "gep") if clave == "C_gep" else (clave, "tree")
            for seed in range(n):
                mae = e * (1 - efecto) + rng.normal(0, 0.01 * e) if clave != "baseline" else e
                if clave == "A":
                    mae = e * (1 + abs(rng.normal(0, 0.002)))  # A no mejora
                hist = list(np.linspace(1.1 * e, 1.0 * e * (1 - efecto), 11)) if conf != "baseline" else []
                columnas = n_attr - 1 if clave in ("A", "C_tree", "C_gep") else n_attr + 2
                registro = {
                    "meta": {"experiment": f"final_{clave}", "dataset": ds, "configuration": conf,
                             "representation": rep, "seed": seed, "config_hash": "h"},
                    "baseline_mae": e, "mae": float(mae), "relative_improvement": float((e - mae) / e),
                    "cv_mae_best": float(e * (1 - efecto) * 1.02), "n_columns": columnas,
                    "n_originals_kept": columnas if clave == "A" else columnas - 2,
                    "n_synthesized": 0 if clave in ("baseline", "A") else 2,
                    "kept_originals": nombres[: (n_attr - 1 if clave in ("A", "C_tree", "C_gep") else n_attr)],
                    "formulas": [] if clave in ("baseline", "A") else
                    [f"log(1 + |({nombres[0]} * {nombres[1]})|)", f"({nombres[2]} / sqrt(|{nombres[3]}|))"],
                    "secondary": {m: {"baseline_mae": e * 0.9, "mae": float(e * 0.9 * (1 - efecto / 2))}
                                  for m in ("ridge", "knn", "rf")},
                    "history": [{"generation": g, "best_fitness": float(h)} for g, h in enumerate(hist)],
                    "distinct_evaluations": 4000 if clave == "C_tree" else 6000, "cache_hits": 10,
                    "seconds": 100.0, "seconds_fit": 90.0,
                }
                ruta = dir_runs / f"final_{clave}" / ds / f"seed_{seed}.json"
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_text(json.dumps(registro))


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("runs")
    _escribir_runs(d)
    return d


# --- piezas estadísticas ------------------------------------------------------------------------------

def test_holm_valores_conocidos():
    aj = an.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert aj["a"] == pytest.approx(0.03) and aj["c"] == pytest.approx(0.06) and aj["b"] == pytest.approx(0.06)
    assert all(0 <= p <= 1 for p in an.holm({"x": 0.9, "y": 0.8}).values())
    assert an.holm({"x": 0.4})["x"] == pytest.approx(0.4)


def test_wilcoxon_unilateral_y_direccion():
    rng = np.random.default_rng(1)
    mejora = -np.abs(rng.normal(1, 0.3, 30))
    assert an.wilcoxon_menor(mejora) < 0.001
    assert an.wilcoxon_menor(-mejora) > 0.99  # empeora: p cercano a 1 en el test unilateral de mejora
    assert an.wilcoxon_menor(np.zeros(30)) == 1.0
    assert an.wilcoxon_menor(mejora) == pytest.approx(stats.wilcoxon(mejora, alternative="less").pvalue)


def test_mann_whitney_direccion_y_constantes():
    x, y = np.arange(30.0), np.arange(30.0) + 10
    assert an.mann_whitney(x, y, "less")["p"] < 0.001  # x < y
    assert an.mann_whitney(x, y, "greater")["p"] > 0.99
    assert an.mann_whitney(np.ones(5), np.ones(5))["p"] == 1.0


def test_gen90_y_cuartiles():
    assert an.gen90(np.array([10.0, 9, 5, 1, 1])) == 3  # 90 % de 9 = 8,1 -> primera g con 10 − f ≥ 8,1
    assert an.gen90(np.full(5, 3.0)) == 0
    assert an.cuartiles(np.arange(1.0, 6.0)) == (2.0, 3.0, 4.0)


def test_nodos_y_operadores_en_infijo():
    assert an.nodos("log(1 + |(a * b)|)", ["a", "b", "c"]) == 4  # 2 variables + mul + log (el + de log no cuenta)
    assert an.nodos("((a + b) / sqrt(|c|))", ["a", "b", "c"]) == 6
    assert an.nodos("a", ["a"]) == 1
    c = an.conteo_operadores("((a + b) - log(1 + |(a * a)|))")
    assert (c["add"], c["sub"], c["mul"], c["div"], c["sqrt"], c["log"]) == (1, 1, 1, 0, 0, 1)
    # los tamaños reales de las fórmulas del proyecto caben en 15 nodos
    from evofeat.formula import size
    from evofeat.repr_gep import GepRepresentation
    from evofeat.formula import to_infix
    rep = GepRepresentation(3, 7, ("add", "sub", "mul", "div", "sqrt", "log"))
    rng = np.random.default_rng(3)
    for _ in range(300):
        f = rep.to_formula(rep.random_genome(rng))
        assert an.nodos(to_infix(f, ["a", "b", "c"]), ["a", "b", "c"]) == size(f)


# --- análisis completo sobre datos con efectos conocidos ----------------------------------------------------

def test_carga_agrupa_por_dataset_y_configuracion(runs):
    g = an.cargar(runs)
    assert set(g) == {"diabetes", "california"}
    assert set(g["diabetes"]) == set(an.CLAVES)
    assert all(len(rs) == 30 for porclave in g.values() for rs in porclave.values())
    assert [r["meta"]["seed"] for r in g["diabetes"]["A"]] == list(range(30))
    with pytest.raises(FileNotFoundError):
        an.cargar(runs / "no_existe")


def test_descriptivo_coincide_con_numpy(runs):
    g = an.cargar(runs)
    d = an.analizar(g)["descriptive"]["california"]["C_tree"]
    ep = np.array([r["mae"] for r in g["california"]["C_tree"]])
    assert d["e_prime_median"] == pytest.approx(np.median(ep))
    assert d["e_prime_q1"] == pytest.approx(np.percentile(ep, 25)) and d["e_prime_q3"] == pytest.approx(np.percentile(ep, 75))
    assert d["e_prime_iqr"] == pytest.approx(d["e_prime_q3"] - d["e_prime_q1"])
    assert d["success_pct"] == pytest.approx(100 * np.mean(ep < 0.5))
    assert d["n"] == 30 and d["n_columns_median"] == 7
    assert d["gap_median"] == pytest.approx(np.median(ep - np.array([r["cv_mae_best"] for r in g["california"]["C_tree"]])))


def test_el_baseline_no_cuenta_como_exito_por_ruido_de_coma_flotante(runs):
    s = an.analizar(an.cargar(runs))
    assert s["descriptive"]["diabetes"]["baseline"]["success_pct"] == 0.0


def test_tests_detectan_los_efectos_sembrados(runs):
    s = an.analizar(an.cargar(runs))
    w = s["wilcoxon_vs_baseline"]
    for ds in DATASETS:
        assert w[ds]["C_tree"]["significant"] and w[ds]["B"]["significant"] and not w[ds]["A"]["significant"]
        assert s["kruskal_A_B_C"][ds]["significant"]
        pares = s["kruskal_A_B_C"][ds]["pairwise"]
        assert pares["A-B"]["significant"] and pares["A-C"]["significant"] and pares["B-C"]["significant"]
        assert pares["A-B"]["p_holm"] >= pares["A-B"]["p_raw"]  # Holm nunca baja el p
        assert s["H3_B_vs_A"][ds]["significant"]
        assert not s["H4_tree_vs_gep"][ds]["e_prime"]["significant"]  # mismo efecto sembrado
        assert s["H4_tree_vs_gep"][ds]["distinct_evaluations"]["significant"]  # 4000 frente a 6000
    h5 = s["H5_B_vs_C"]["california"]
    assert h5["significant"] and h5["median_cost_of_limit"] > 0 and h5["fraction_C_using_n_minus_1_columns"] == 1.0
    assert h5["n_minus_1"] == 7


def test_veredictos(runs):
    v = an.analizar(an.cargar(runs))["verdicts"]
    assert v["H1"]["supported"] is True
    assert v["H2"]["supported"] is None  # sin umbral en el diseño: interpretación manual
    assert v["H3"]["supported"] is True and v["H4"]["supported"] is True and v["H5"]["supported"] is True
    assert "no rechazar" in v["H4"]["note"]


def test_veredictos_cambian_si_los_efectos_cambian(tmp_path):
    # B no mejora más que A, C empeora a B: H3 y H5 no se sostienen
    _escribir_runs(tmp_path, {"baseline": 0.0, "A": 0.0, "B": 0.0, "C_tree": 0.0, "C_gep": 0.0})
    v = an.analizar(an.cargar(tmp_path))["verdicts"]
    assert v["H1"]["supported"] is False and v["H3"]["supported"] is False and v["H5"]["supported"] is False


def test_secundarios_y_formulas(runs):
    g = an.cargar(runs)
    s = an.analizar(g)
    sec = s["secondary"]["california"]["C_tree"]
    assert set(sec) == {"ridge", "knn", "rf"} and sec["ridge"]["success_pct"] == 100.0
    f = an.analisis_formulas(g)["diabetes"]["C_tree"]
    assert f["formulas_total"] == 60 and f["distinct_formulas"] == 2
    assert f["top_formulas"][0][1] == 30
    nombres = list(f["kept_fraction"])
    assert f["kept_fraction"][nombres[0]] == 1.0 and f["kept_fraction"][nombres[-1]] == 0.0  # el último se elimina siempre
    assert f["operator_counts"]["mul"] == 30 and f["operator_counts"]["log"] == 30 and f["operator_counts"]["sqrt"] == 30
    assert f["variable_use_fraction"][nombres[0]] == 1.0


def test_ejecutar_escribe_tablas_estadistica_y_figuras(runs, tmp_path):
    out = tmp_path / "analysis"
    an.ejecutar(runs, out)
    for rel in ("informe.md", "estadistica.json", "formulas.json", "tablas/descriptivo.md", "tablas/descriptivo.csv",
                "tablas/modelos_secundarios.md", "tablas/modelos_secundarios.csv", "tablas/estadistica.md",
                "tablas/formulas.md", "figuras/cajas_e_prime.png", "figuras/convergencia.png",
                "figuras/convergencia_arbol_vs_gep.png", "figuras/originales_conservados.png"):
        assert (out / rel).exists() and (out / rel).stat().st_size > 0, rel
    json.loads((out / "estadistica.json").read_text())  # JSON válido, sin NaN (allow_nan=False al escribir)
    informe = (out / "informe.md").read_text()
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert f"**{h}**" in informe
    # CSV: una fila por dataset × configuración
    lineas = (out / "tablas" / "descriptivo.csv").read_text().strip().splitlines()
    assert len(lineas) == 1 + 2 * 5
    assert (out / "figuras" / "cajas_e_prime.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- contra los resultados reales del repositorio -----------------------------------------------------------------

@pytest.mark.skipif(not list((RAIZ / "results" / "runs").glob("*/*/seed_*.json")), reason="no hay resultados reales")
def test_analisis_sobre_los_resultados_reales_coincide_con_los_json():
    g = an.cargar(RAIZ / "results" / "runs")
    s = an.analizar(g)
    for ds in g:
        for clave, rs in g[ds].items():
            assert len(rs) == 30
            ep = np.array([r["mae"] for r in rs])
            assert s["descriptive"][ds][clave]["e_prime_median"] == pytest.approx(float(np.median(ep)))
        # e es un valor fijo con OLS: igual en todas las ejecuciones y configuraciones
        es = {r["baseline_mae"] for rs in g[ds].values() for r in rs}
        assert max(es) - min(es) < 1e-9
    json.dumps(s, allow_nan=False)
