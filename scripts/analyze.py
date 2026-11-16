"""Análisis de resultados (hito 10): tablas, tests estadísticos y figuras del §11.

Implementa RF-12, D22, D25 y §11. Lee todos los JSON de ``results/runs`` y escribe, en
``results/analysis`` (solo el código escribe aquí; nada se edita a mano):

- ``tablas/descriptivo.{md,csv}``: por dataset y configuración, e, mediana y rango intercuartílico
  de e′, % de semillas con e′ < e, mejora relativa, columnas, sobreajuste (e′ − MAE en CV) y coste.
- ``tablas/modelos_secundarios.{md,csv}``: e y e′ con Ridge, KNN y Random Forest (D11).
- ``tablas/formulas.md``: fórmulas más frecuentes, uso de originales y operadores, y originales
  más eliminados (§11).
- ``estadistica.json`` y ``tablas/estadistica.md``: tests de D22 y D25 y contraste de H1–H5.
- ``figuras/*.png``: diagramas de caja de e′, curvas de convergencia y frecuencia de los originales.
- ``informe.md``: todo lo anterior en un solo documento de lectura.

Tests (D22): Wilcoxon de una muestra sobre e′ − e (unilateral, α = 0,05) frente a la línea base;
Kruskal-Wallis entre A, B y C (árbol) y, si hay diferencias, Mann-Whitney por pares con corrección
de Holm; H3: B frente a A (unilateral); H4/D25: C con árboles frente a C con GEP (bilateral);
H5: B frente a C en California (unilateral). No se corrige entre hipótesis distintas. No hay
veredicto automático para H2 («pequeña o nula» no tiene umbral en el diseño): se dan las cifras.

Medida propia de velocidad de convergencia (D25, no fijada por el diseño): generación en la que se
alcanza el 90 % de la mejora total del mejor fitness en CV (``gen90``).

Uso: python scripts/analyze.py results/runs [--out results/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from evofeat import data  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ALPHA = 0.05
TOL_MEJORA = 1e-9  # e′ < e solo cuenta si la mejora supera el ruido de coma flotante (relativo a e)

CLAVES = ("baseline", "A", "B", "C_tree", "C_gep")
ETIQUETA = {"baseline": "Línea base", "A": "A (selección)", "B": "B (síntesis)",
            "C_tree": "C (árbol)", "C_gep": "C (GEP)"}
ETIQUETA_CORTA = {"baseline": "Línea base", "A": "A", "B": "B", "C_tree": "C árbol", "C_gep": "C GEP"}
# Paleta categórica validada (blue, orange, aqua, yellow; slots 1–4). Contraste bajo en aqua y
# amarillo sobre fondo claro: se compensa con etiquetas directas, leyenda y tablas.
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C_tree": "#1baf7a", "C_gep": "#eda100"}
GRIS = "#52514e"
ESTILO = {"A": "-", "B": "--", "C_tree": "-", "C_gep": "-."}
MODELOS_SEC = ("ridge", "knn", "rf")
NOMBRE_MODELO = {"ridge": "Ridge", "knn": "KNN", "rf": "Random Forest"}
NOMBRE_DATASET = {"diabetes": "Diabetes", "california": "California"}
OPERADORES = ("add", "sub", "mul", "div", "sqrt", "log")


# ---------------------------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------------------------

def clave_de(meta: dict[str, Any]) -> str:
    """Clave de configuración: baseline, A, B, C_tree o C_gep."""
    c = meta["configuration"]
    if c == "C":
        return "C_tree" if meta["representation"] == "tree" else "C_gep"
    return c


def cargar(dir_runs: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """dataset -> clave -> ejecuciones ordenadas por semilla."""
    grupos: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for f in sorted(Path(dir_runs).glob("*/*/seed_*.json")):
        r = json.loads(f.read_text())
        grupos.setdefault(r["meta"]["dataset"], {}).setdefault(clave_de(r["meta"]), []).append(r)
    for porclave in grupos.values():
        for rs in porclave.values():
            rs.sort(key=lambda r: r["meta"]["seed"])
    if not grupos:
        raise FileNotFoundError(f"no hay resultados en {dir_runs}")
    return grupos


def v(rs: list[dict[str, Any]], campo: str) -> np.ndarray:
    return np.array([r[campo] for r in rs], dtype=float)


# ---------------------------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------------------------

def wilcoxon_menor(d: np.ndarray) -> float:
    """p de Wilcoxon de una muestra, unilateral (H1: mediana de d < 0). 1,0 si todas las d son 0."""
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="less", zero_method="wilcox").pvalue)


def mann_whitney(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> dict[str, float]:
    """Mann-Whitney U (x frente a y). Si todos los valores son iguales, p = 1."""
    if np.ptp(np.concatenate([x, y])) == 0:
        return {"U": float(len(x) * len(y) / 2), "p": 1.0}
    r = stats.mannwhitneyu(x, y, alternative=alternative)
    return {"U": float(r.statistic), "p": float(r.pvalue)}


def holm(pvalores: dict[str, float]) -> dict[str, float]:
    """Corrección de Holm-Bonferroni: devuelve los p ajustados."""
    orden = sorted(pvalores, key=lambda k: pvalores[k])
    m, previo, aj = len(orden), 0.0, {}
    for i, k in enumerate(orden):
        previo = max(previo, min(1.0, (m - i) * pvalores[k]))
        aj[k] = previo
    return aj


def cuartiles(x: np.ndarray) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(q1), float(med), float(q3)


def gen90(mejor: np.ndarray) -> int:
    """Primera generación con el 90 % de la mejora total del mejor fitness en CV."""
    total = mejor[0] - mejor[-1]
    if total <= 0:
        return 0
    return int(np.argmax(mejor[0] - mejor >= 0.9 * total))


def historial(rs: list[dict[str, Any]]) -> np.ndarray:
    """(ejecuciones × generaciones) del mejor fitness en CV."""
    return np.array([[h["best_fitness"] for h in r["history"]] for r in rs], dtype=float)


# ---------------------------------------------------------------------------------------------
# Fórmulas (RNF-04: se analizan en notación infija)
# ---------------------------------------------------------------------------------------------

def _patron_variables(nombres: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(nombres, key=len, reverse=True)) + r")\b")


def conteo_operadores(texto: str) -> Counter[str]:
    """Cuenta los operadores de una fórmula en infijo (+, −, ×, ÷ binarios, √|·| y log(1+|·|))."""
    n_log = texto.count("log(1 + |")
    return Counter({
        "add": texto.count(" + ") - n_log,  # el " + " de log(1 + |x|) no es una suma binaria
        "sub": texto.count(" - "), "mul": texto.count(" * "), "div": texto.count(" / "),
        "sqrt": texto.count("sqrt(|"), "log": n_log,
    })


def nodos(texto: str, nombres: list[str]) -> int:
    """Número de nodos de una fórmula en infijo: variables + operadores."""
    return len(_patron_variables(nombres).findall(texto)) + sum(conteo_operadores(texto).values())


# ---------------------------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------------------------

def descriptivo(rs: list[dict[str, Any]]) -> dict[str, float]:
    e, ep = v(rs, "baseline_mae"), v(rs, "mae")
    q1, med, q3 = cuartiles(ep)
    brecha = ep - v(rs, "cv_mae_best") if all(r.get("cv_mae_best") is not None for r in rs) else np.full(len(rs), np.nan)
    return {
        "n": len(rs), "e": float(np.median(e)), "e_prime_median": med, "e_prime_q1": q1, "e_prime_q3": q3,
        "e_prime_iqr": q3 - q1, "success_pct": float(100 * np.mean(e - ep > TOL_MEJORA * e)),
        "rel_improvement_median": float(np.median(v(rs, "relative_improvement"))),
        "n_columns_median": float(np.median(v(rs, "n_columns"))),
        "n_synthesized_median": float(np.median(v(rs, "n_synthesized"))),
        "cv_mae_median": float(np.nanmedian([r["cv_mae_best"] if r["cv_mae_best"] is not None else np.nan for r in rs])),
        "gap_median": float(np.nanmedian(brecha)),
        "seconds_median": float(np.median(v(rs, "seconds"))),
        "distinct_evaluations_median": float(np.median(v(rs, "distinct_evaluations"))),
    }


def wilcoxon_frente_a_base(rs: list[dict[str, Any]]) -> dict[str, float]:
    d = v(rs, "mae") - v(rs, "baseline_mae")
    p = wilcoxon_menor(d)
    return {"n": len(d), "median_diff": float(np.median(d)),
            "median_rel_improvement": float(np.median(v(rs, "relative_improvement"))),
            "p_value": p, "significant": bool(p < ALPHA)}


def analizar(grupos: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Todos los números del análisis (D22, D25, H1–H5), sin dibujar nada."""
    out: dict[str, Any] = {"alpha": ALPHA, "descriptive": {}, "wilcoxon_vs_baseline": {}, "kruskal_A_B_C": {},
                           "H3_B_vs_A": {}, "H4_tree_vs_gep": {}, "H5_B_vs_C": {}, "secondary": {}}
    for ds, g in grupos.items():
        out["descriptive"][ds] = {k: descriptivo(g[k]) for k in CLAVES if k in g}
        out["wilcoxon_vs_baseline"][ds] = {k: wilcoxon_frente_a_base(g[k]) for k in ("A", "B", "C_tree", "C_gep") if k in g}
        if all(k in g for k in ("A", "B", "C_tree")):
            grupos_e = [v(g[k], "mae") for k in ("A", "B", "C_tree")]
            if np.ptp(np.concatenate(grupos_e)) == 0:
                kw = {"H": 0.0, "p": 1.0}
            else:
                r = stats.kruskal(*grupos_e)
                kw = {"H": float(r.statistic), "p": float(r.pvalue)}
            kw["significant"] = bool(kw["p"] < ALPHA)
            pares = {"A-B": ("A", "B"), "A-C": ("A", "C_tree"), "B-C": ("B", "C_tree")}
            crudos = {n: mann_whitney(v(g[a], "mae"), v(g[b], "mae"))["p"] for n, (a, b) in pares.items()}
            ajust = holm(crudos)
            kw["pairwise"] = {n: {"p_raw": crudos[n], "p_holm": ajust[n], "significant": bool(ajust[n] < ALPHA),
                                  "median_a": float(np.median(v(g[pares[n][0]], "mae"))),
                                  "median_b": float(np.median(v(g[pares[n][1]], "mae")))} for n in pares}
            kw["pairwise_computed_only_if_kruskal_significant"] = kw["significant"]
            out["kruskal_A_B_C"][ds] = kw
        if "A" in g and "B" in g:
            r = mann_whitney(v(g["B"], "mae"), v(g["A"], "mae"), alternative="less")  # B < A
            out["H3_B_vs_A"][ds] = {**r, "median_B": float(np.median(v(g["B"], "mae"))),
                                    "median_A": float(np.median(v(g["A"], "mae"))), "significant": bool(r["p"] < ALPHA)}
        if "C_tree" in g and "C_gep" in g:
            out["H4_tree_vs_gep"][ds] = _comparar_representaciones(ds, g["C_tree"], g["C_gep"])
        if "B" in g and "C_tree" in g:
            r = mann_whitney(v(g["B"], "mae"), v(g["C_tree"], "mae"), alternative="less")  # B < C
            n = len(data.load_raw(ds).nombres)
            frac = float(np.mean(v(g["C_tree"], "n_columns") == n - 1))
            out["H5_B_vs_C"][ds] = {**r, "median_B": float(np.median(v(g["B"], "mae"))),
                                    "median_C": float(np.median(v(g["C_tree"], "mae"))),
                                    "median_cost_of_limit": float(np.median(v(g["C_tree"], "mae")) - np.median(v(g["B"], "mae"))),
                                    "fraction_C_using_n_minus_1_columns": frac, "n_minus_1": n - 1,
                                    "significant": bool(r["p"] < ALPHA)}
        out["secondary"][ds] = _secundarios(g)
    out["verdicts"] = veredictos(out)
    return out


def _comparar_representaciones(ds: str, arbol: list[dict[str, Any]], gep: list[dict[str, Any]]) -> dict[str, Any]:
    nombres = data.load_raw(ds).nombres

    def tamano(rs: list[dict[str, Any]]) -> np.ndarray:  # nodos totales de las fórmulas de cada ejecución
        return np.array([sum(nodos(f, nombres) for f in r["formulas"]) for r in rs], dtype=float)

    def conv(rs: list[dict[str, Any]]) -> np.ndarray:
        return np.array([gen90(h) for h in historial(rs)], dtype=float)

    medidas = {
        "e_prime": (v(arbol, "mae"), v(gep, "mae")),
        "n_columns": (v(arbol, "n_columns"), v(gep, "n_columns")),
        "formula_nodes_total": (tamano(arbol), tamano(gep)),
        "gen90": (conv(arbol), conv(gep)),
        "seconds": (v(arbol, "seconds"), v(gep, "seconds")),
        "distinct_evaluations": (v(arbol, "distinct_evaluations"), v(gep, "distinct_evaluations")),
    }
    res: dict[str, Any] = {}
    for nombre, (a, b) in medidas.items():
        r = mann_whitney(a, b)
        res[nombre] = {"median_tree": float(np.median(a)), "median_gep": float(np.median(b)),
                       "U": r["U"], "p": r["p"], "significant": bool(r["p"] < ALPHA)}
    return res


def _secundarios(g: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("A", "B", "C_tree", "C_gep"):
        if k not in g:
            continue
        out[k] = {}
        for m in MODELOS_SEC:
            e = np.array([r["secondary"][m]["baseline_mae"] for r in g[k]])
            ep = np.array([r["secondary"][m]["mae"] for r in g[k]])
            q1, med, q3 = cuartiles(ep)
            out[k][m] = {"e_median": float(np.median(e)), "e_prime_median": med, "e_prime_q1": q1, "e_prime_q3": q3,
                         "success_pct": float(100 * np.mean(e - ep > TOL_MEJORA * e)),
                         "rel_improvement_median": float(np.median((e - ep) / e)),
                         "p_value": wilcoxon_menor(ep - e)}
    return out


def veredictos(s: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Contraste de H1–H5 con los tests anteriores. H2 no tiene veredicto automático."""
    v_: dict[str, dict[str, Any]] = {}
    w = s["wilcoxon_vs_baseline"]
    if "california" in w and "C_tree" in w["california"]:
        x = w["california"]["C_tree"]
        v_["H1"] = {"text": "En California, C mejora claramente (e′ < e).",
                    "supported": bool(x["significant"] and x["median_rel_improvement"] > 0),
                    "evidence": f"mejora relativa mediana {100 * x['median_rel_improvement']:.2f} %, Wilcoxon unilateral p = {x['p_value']:.3g}"}
    if "diabetes" in w and "C_tree" in w["diabetes"]:
        x = w["diabetes"]["C_tree"]
        v_["H2"] = {"text": "En Diabetes, la mejora es pequeña o nula.", "supported": None,
                    "evidence": f"mejora relativa mediana {100 * x['median_rel_improvement']:.2f} %, Wilcoxon unilateral p = {x['p_value']:.3g}; "
                                "el diseño no fija un umbral para «pequeña»: interpretación manual"}
    h3 = s["H3_B_vs_A"]
    if h3:
        v_["H3"] = {"text": "B aporta más que A en ambos datasets.",
                    "supported": bool(all(x["significant"] for x in h3.values()) and len(h3) == 2),
                    "evidence": "; ".join(f"{NOMBRE_DATASET.get(d, d)}: mediana e′ B = {x['median_B']:.4g}, A = {x['median_A']:.4g}, p = {x['p']:.3g}" for d, x in h3.items())}
    h4 = s["H4_tree_vs_gep"]
    if h4:
        v_["H4"] = {"text": "Sin diferencia significativa en e′ entre árboles y GEP; si la hay, en la velocidad de convergencia.",
                    "supported": bool(all(not x["e_prime"]["significant"] for x in h4.values())),
                    "evidence": "; ".join(f"{NOMBRE_DATASET.get(d, d)}: e′ árbol = {x['e_prime']['median_tree']:.4g}, GEP = {x['e_prime']['median_gep']:.4g}, p = {x['e_prime']['p']:.3g}; "
                                          f"gen90 árbol = {x['gen90']['median_tree']:.0f}, GEP = {x['gen90']['median_gep']:.0f}, p = {x['gen90']['p']:.3g}" for d, x in h4.items()),
                    "note": "no rechazar la igualdad no prueba que sea cierta"}
    h5 = s["H5_B_vs_C"].get("california")
    if h5:
        v_["H5"] = {"text": "En California, B obtiene un e′ menor que C (coste de exigir s < n).",
                    "supported": bool(h5["significant"] and h5["fraction_C_using_n_minus_1_columns"] > 0),
                    "evidence": f"mediana e′ B = {h5['median_B']:.4g}, C = {h5['median_C']:.4g} (coste {h5['median_cost_of_limit']:+.4g}), p = {h5['p']:.3g}; "
                                f"C usa n − 1 = {h5['n_minus_1']} columnas en el {100 * h5['fraction_C_using_n_minus_1_columns']:.0f} % de las ejecuciones"}
    return v_


def analisis_formulas(grupos: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Fórmulas más frecuentes, uso de originales y operadores, y originales conservados."""
    out: dict[str, Any] = {}
    for ds, g in grupos.items():
        nombres = data.load_raw(ds).nombres
        pat = _patron_variables(nombres)
        out[ds] = {}
        for k in ("A", "B", "C_tree", "C_gep"):
            if k not in g:
                continue
            formulas = [f for r in g[k] for f in r["formulas"]]
            uso_vars = Counter(n for r in g[k] for n in set(pat.findall(" ".join(r["formulas"]))))
            ops = Counter()
            for f in formulas:
                ops += conteo_operadores(f)
            conservados = Counter(n for r in g[k] for n in r["kept_originals"])
            n_runs = len(g[k])
            out[ds][k] = {
                "runs": n_runs, "formulas_total": len(formulas), "distinct_formulas": len(set(formulas)),
                "top_formulas": Counter(formulas).most_common(5),
                "variable_use_fraction": {n: uso_vars.get(n, 0) / n_runs for n in nombres},
                "operator_counts": {o: ops.get(o, 0) for o in OPERADORES},
                "kept_fraction": {n: conservados.get(n, 0) / n_runs for n in nombres},
                "formula_nodes_median": float(np.median([nodos(f, nombres) for f in formulas])) if formulas else 0.0,
                "formula_nodes_max": int(max((nodos(f, nombres) for f in formulas), default=0)),
            }
    return out


# ---------------------------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------------------------

def num(x: float, cifras: int = 4) -> str:
    return f"{x:.{cifras}g}"


def md_tabla(cabecera: list[str], filas: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(cabecera) + " |", "| " + " | ".join("---" for _ in cabecera) + " |"]
                     + ["| " + " | ".join(f) + " |" for f in filas])


def escribir_csv(ruta: Path, cabecera: list[str], filas: list[list[Any]]) -> None:
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)


def tablas_descriptivo(s: dict[str, Any], dir_tablas: Path) -> str:
    cab = ["Dataset", "Configuración", "n", "e", "mediana e′", "Q1", "Q3", "RIC", "% e′ < e", "mejora rel. mediana",
           "columnas", "sintetizadas", "MAE CV mediana", "e′ − MAE CV", "tiempo (s)", "eval. distintas"]
    filas, csv_f = [], []
    for ds, porclave in s["descriptive"].items():
        for k, d in porclave.items():
            fila = [NOMBRE_DATASET.get(ds, ds), ETIQUETA[k], str(d["n"]), num(d["e"]), num(d["e_prime_median"]),
                    num(d["e_prime_q1"]), num(d["e_prime_q3"]), num(d["e_prime_iqr"]), f"{d['success_pct']:.0f}",
                    f"{100 * d['rel_improvement_median']:.2f} %", f"{d['n_columns_median']:g}", f"{d['n_synthesized_median']:g}",
                    num(d["cv_mae_median"]), num(d["gap_median"]), f"{d['seconds_median']:.0f}", f"{d['distinct_evaluations_median']:.0f}"]
            filas.append(fila)
            csv_f.append([ds, k, d["n"], d["e"], d["e_prime_median"], d["e_prime_q1"], d["e_prime_q3"], d["e_prime_iqr"],
                          d["success_pct"], d["rel_improvement_median"], d["n_columns_median"], d["n_synthesized_median"],
                          d["cv_mae_median"], d["gap_median"], d["seconds_median"], d["distinct_evaluations_median"]])
    escribir_csv(dir_tablas / "descriptivo.csv", ["dataset", "config", "n", "e", "e_prime_median", "q1", "q3", "iqr", "success_pct",
                                                  "rel_improvement_median", "n_columns_median", "n_synthesized_median",
                                                  "cv_mae_median", "gap_median", "seconds_median", "distinct_evaluations_median"], csv_f)
    texto = md_tabla(cab, filas)
    (dir_tablas / "descriptivo.md").write_text(texto + "\n")
    return texto


def tabla_secundarios(s: dict[str, Any], dir_tablas: Path) -> str:
    cab = ["Dataset", "Configuración", "Modelo", "mediana e", "mediana e′", "Q1", "Q3", "% e′ < e", "mejora rel. mediana", "p (Wilcoxon)"]
    filas, csv_f = [], []
    for ds, porclave in s["secondary"].items():
        for k, porm in porclave.items():
            for m, d in porm.items():
                filas.append([NOMBRE_DATASET.get(ds, ds), ETIQUETA[k], NOMBRE_MODELO[m], num(d["e_median"]), num(d["e_prime_median"]),
                              num(d["e_prime_q1"]), num(d["e_prime_q3"]), f"{d['success_pct']:.0f}",
                              f"{100 * d['rel_improvement_median']:.2f} %", num(d["p_value"], 3)])
                csv_f.append([ds, k, m, d["e_median"], d["e_prime_median"], d["e_prime_q1"], d["e_prime_q3"], d["success_pct"],
                              d["rel_improvement_median"], d["p_value"]])
    escribir_csv(dir_tablas / "modelos_secundarios.csv", ["dataset", "config", "model", "e_median", "e_prime_median", "q1", "q3",
                                                          "success_pct", "rel_improvement_median", "p_wilcoxon"], csv_f)
    texto = md_tabla(cab, filas)
    (dir_tablas / "modelos_secundarios.md").write_text(texto + "\n")
    return texto


def tabla_estadistica(s: dict[str, Any], dir_tablas: Path) -> str:
    partes = [f"Nivel de significación α = {s['alpha']}. Los p-valores no se corrigen entre hipótesis distintas."]
    partes += ["", "### Wilcoxon de una muestra sobre e′ − e (unilateral: e′ < e)", ""]
    filas = [[NOMBRE_DATASET.get(ds, ds), ETIQUETA[k], num(x["median_diff"]), f"{100 * x['median_rel_improvement']:.2f} %",
              num(x["p_value"], 3), "sí" if x["significant"] else "no"]
             for ds, porclave in s["wilcoxon_vs_baseline"].items() for k, x in porclave.items()]
    partes.append(md_tabla(["Dataset", "Configuración", "mediana e′ − e", "mejora rel. mediana", "p", "significativo"], filas))
    partes += ["", "### Kruskal-Wallis entre A, B y C (árbol) sobre e′, y Mann-Whitney por pares con Holm", ""]
    filas = []
    for ds, kw in s["kruskal_A_B_C"].items():
        filas.append([NOMBRE_DATASET.get(ds, ds), "Kruskal-Wallis", f"H = {kw['H']:.3g}", "", num(kw["p"], 3), "", "sí" if kw["significant"] else "no"])
        for par, x in kw["pairwise"].items():
            filas.append([NOMBRE_DATASET.get(ds, ds), f"Mann-Whitney {par}", f"{num(x['median_a'])} vs {num(x['median_b'])}",
                          num(x["p_raw"], 3), num(x["p_holm"], 3), "", "sí" if x["significant"] else "no"])
    partes.append(md_tabla(["Dataset", "Contraste", "medianas / estadístico", "p crudo", "p (Holm)", "", "significativo"], filas))
    partes.append("\nLos pares solo son interpretables si Kruskal-Wallis es significativo (D22).")
    partes += ["", "### H3: B frente a A (unilateral: e′ de B < e′ de A)", ""]
    partes.append(md_tabla(["Dataset", "mediana e′ B", "mediana e′ A", "p", "significativo"],
                           [[NOMBRE_DATASET.get(d, d), num(x["median_B"]), num(x["median_A"]), num(x["p"], 3), "sí" if x["significant"] else "no"]
                            for d, x in s["H3_B_vs_A"].items()]))
    partes += ["", "### H4 (D25): C con árboles frente a C con GEP (bilateral)", ""]
    nombres_m = {"e_prime": "e′", "n_columns": "columnas", "formula_nodes_total": "nodos de fórmula (total)",
                 "gen90": "generación al 90 % de la mejora", "seconds": "tiempo (s)", "distinct_evaluations": "evaluaciones distintas"}
    filas = [[NOMBRE_DATASET.get(d, d), nombres_m[m], num(x["median_tree"]), num(x["median_gep"]), num(x["p"], 3), "sí" if x["significant"] else "no"]
             for d, porm in s["H4_tree_vs_gep"].items() for m, x in porm.items()]
    partes.append(md_tabla(["Dataset", "medida", "mediana árbol", "mediana GEP", "p", "significativo"], filas))
    partes += ["", "### H5: B frente a C en California (unilateral: e′ de B < e′ de C)", ""]
    partes.append(md_tabla(["Dataset", "mediana e′ B", "mediana e′ C", "coste de s < n (C − B)", "p", "C usa n − 1 columnas", "significativo"],
                           [[NOMBRE_DATASET.get(d, d), num(x["median_B"]), num(x["median_C"]), f"{x['median_cost_of_limit']:+.4g}", num(x["p"], 3),
                             f"{100 * x['fraction_C_using_n_minus_1_columns']:.0f} %", "sí" if x["significant"] else "no"]
                            for d, x in s["H5_B_vs_C"].items()]))
    texto = "\n".join(partes)
    (dir_tablas / "estadistica.md").write_text(texto + "\n")
    return texto


def tabla_formulas(f: dict[str, Any], dir_tablas: Path) -> str:
    partes = []
    for ds, porclave in f.items():
        partes += [f"### {NOMBRE_DATASET.get(ds, ds)}", ""]
        nombres = list(next(iter(porclave.values()))["kept_fraction"])
        partes.append("Fracción de ejecuciones que conservan cada atributo original (los que menos aparecen son los más eliminados); B conserva todos por construcción:\n")
        partes.append(md_tabla(["Atributo"] + [ETIQUETA_CORTA[k] for k in porclave if k != "B"],
                               [[n] + [f"{porclave[k]['kept_fraction'][n]:.2f}" for k in porclave if k != "B"] for n in nombres]))
        partes.append("\nFracción de ejecuciones cuyas fórmulas usan cada atributo original:\n")
        claves_f = [k for k in porclave if porclave[k]["formulas_total"] > 0]
        partes.append(md_tabla(["Atributo"] + [ETIQUETA_CORTA[k] for k in claves_f],
                               [[n] + [f"{porclave[k]['variable_use_fraction'][n]:.2f}" for k in claves_f] for n in nombres]))
        partes.append("\nOperadores en las fórmulas finales (recuento total):\n")
        partes.append(md_tabla(["Operador"] + [ETIQUETA_CORTA[k] for k in claves_f],
                               [[o] + [str(porclave[k]["operator_counts"][o]) for k in claves_f] for o in OPERADORES]))
        partes.append("\nTamaño de las fórmulas y diversidad:\n")
        partes.append(md_tabla(["Configuración", "fórmulas", "distintas", "nodos (mediana)", "nodos (máx.)"],
                               [[ETIQUETA_CORTA[k], str(porclave[k]["formulas_total"]), str(porclave[k]["distinct_formulas"]),
                                 f"{porclave[k]['formula_nodes_median']:g}", str(porclave[k]["formula_nodes_max"])] for k in claves_f]))
        for k in claves_f:
            partes.append(f"\nFórmulas más frecuentes — {ETIQUETA[k]}:\n")
            partes += [f"- ({c}×) `{t}`" for t, c in porclave[k]["top_formulas"]]
        partes.append("")
    texto = "\n".join(partes)
    (dir_tablas / "formulas.md").write_text(texto + "\n")
    return texto


# ---------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------

def _estilo(ax: plt.Axes) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GRIS, labelsize=9)


def fig_cajas(grupos: dict[str, dict[str, list[dict[str, Any]]]], ruta: Path) -> None:
    dss = [d for d in ("diabetes", "california") if d in grupos] or list(grupos)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.8 * len(dss), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = grupos[ds]
        claves = [k for k in ("A", "B", "C_tree", "C_gep") if k in g]
        bp = ax.boxplot([v(g[k], "mae") for k in claves], positions=range(len(claves)), widths=0.5, patch_artist=True,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.5}, whiskerprops={"color": GRIS, "linewidth": 1},
                        capprops={"color": GRIS, "linewidth": 1}, flierprops={"marker": "o", "markersize": 3, "markeredgecolor": GRIS})
        for caja, k in zip(bp["boxes"], claves):
            caja.set(facecolor=COLOR[k], alpha=0.55, edgecolor=COLOR[k], linewidth=1.2)
        e = float(np.median(v(g["baseline"], "baseline_mae"))) if "baseline" in g else float(np.median(v(g[claves[0]], "baseline_mae")))
        ax.axhline(e, color=GRIS, linewidth=1.4, linestyle="--")
        # escala: bigotes y e; los valores atípicos más allá se cuentan en lugar de aplastar las cajas
        bigotes = [float(w.get_ydata()[1]) for w in bp["whiskers"]]
        bajo = min([e] + [float(w.get_ydata()[1]) for w in bp["whiskers"]])
        techo = max([e] + bigotes)
        margen = 0.12 * (techo - bajo)
        ax.set_ylim(bajo - margen, techo + margen)
        for i, k in enumerate(claves):
            fuera = v(g[k], "mae")
            n_fuera = int(np.sum(fuera > techo + margen))
            if n_fuera:
                ax.text(i, techo + margen, f"↑ {n_fuera} más\n(máx. {fuera.max():.3g})", ha="center", va="top", fontsize=8, color=GRIS)
        ax.set_xlim(-0.6, len(claves) - 0.4 + 0.55)
        ax.text(len(claves) - 0.4 + 0.5, e, f"e = {e:.4g}", color=GRIS, va="bottom", ha="right", fontsize=9)
        ax.set_xticks(range(len(claves)), [ETIQUETA_CORTA[k] for k in claves])
        ax.set_title(f"{NOMBRE_DATASET.get(ds, ds)} — e′ por configuración (n = {len(g[claves[0]])})", fontsize=11, loc="left")
        ax.set_ylabel("MAE en validación (e′)", color=GRIS)
        _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def fig_convergencia(grupos: dict[str, dict[str, list[dict[str, Any]]]], ruta: Path, claves_ok: tuple[str, ...]) -> None:
    dss = [d for d in ("diabetes", "california") if d in grupos] or list(grupos)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.8 * len(dss), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        g = grupos[ds]
        for k in [c for c in claves_ok if c in g]:
            h = historial(g[k])
            x = np.arange(h.shape[1])
            q1, med, q3 = np.percentile(h, [25, 50, 75], axis=0)
            ax.fill_between(x, q1, q3, color=COLOR[k], alpha=0.15, linewidth=0)
            ax.plot(x, med, color=COLOR[k], linewidth=2, linestyle=ESTILO[k], label=ETIQUETA_CORTA[k])
        ax.set_xlim(0, x[-1])
        ax.set_xlabel("Generación", color=GRIS)
        ax.set_ylabel("Mejor MAE en CV (mediana y RIC entre semillas)", color=GRIS)
        ax.set_title(NOMBRE_DATASET.get(ds, ds), fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
        _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def fig_originales(f: dict[str, Any], ruta: Path) -> None:
    dss = [d for d in ("diabetes", "california") if d in f] or list(f)
    fig, axes = plt.subplots(1, len(dss), figsize=(5.8 * len(dss), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        porclave = f[ds]
        claves = [k for k in ("A", "C_tree", "C_gep") if k in porclave]
        nombres = list(porclave[claves[0]]["kept_fraction"])
        ancho = 0.8 / len(claves)
        for i, k in enumerate(claves):
            xs = np.arange(len(nombres)) + (i - (len(claves) - 1) / 2) * ancho
            ax.bar(xs, [porclave[k]["kept_fraction"][n] for n in nombres], width=ancho * 0.9, color=COLOR[k], label=ETIQUETA_CORTA[k])
        ax.set_xticks(range(len(nombres)), nombres, rotation=35, ha="right")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Fracción de ejecuciones que lo conservan", color=GRIS)
        ax.set_title(f"{NOMBRE_DATASET.get(ds, ds)} — atributos originales conservados", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, ncol=len(claves), loc="upper center")
        _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------------------------

def informe(s: dict[str, Any], t_desc: str, t_sec: str, t_est: str, t_form: str) -> str:
    partes = ["# Resultados", "", "Generado por `scripts/analyze.py` a partir de `results/runs`; no se edita a mano.", "",
              "## Contraste de hipótesis (`docs/HIPOTESIS.md`)", ""]
    for h, x in s["verdicts"].items():
        estado = "interpretación manual" if x["supported"] is None else ("apoyada" if x["supported"] else "no apoyada")
        partes.append(f"- **{h}** — {x['text']} → **{estado}**. {x['evidence']}." + (f" ({x['note']})" if "note" in x else ""))
    partes += ["", "## e frente a e′ por dataset y configuración", "", t_desc, "",
               "## Pruebas estadísticas", "", t_est, "",
               "## Modelos secundarios (D11)", "", t_sec, "",
               "## Fórmulas y atributos", "", t_form, "",
               "## Figuras", "", "- `figuras/cajas_e_prime.png`: diagramas de caja de e′ con la línea de e.",
               "- `figuras/convergencia.png`: mejor MAE en CV por generación (mediana y RIC).",
               "- `figuras/convergencia_arbol_vs_gep.png`: C con árboles frente a C con GEP.",
               "- `figuras/originales_conservados.png`: fracción de ejecuciones que conserva cada original.", ""]
    return "\n".join(partes)


def ejecutar(dir_runs: Path, dir_out: Path) -> dict[str, Any]:
    grupos = cargar(dir_runs)
    (dir_out / "tablas").mkdir(parents=True, exist_ok=True)
    (dir_out / "figuras").mkdir(parents=True, exist_ok=True)
    s = analizar(grupos)
    f = analisis_formulas(grupos)
    (dir_out / "estadistica.json").write_text(json.dumps(s, indent=1, allow_nan=False))
    (dir_out / "formulas.json").write_text(json.dumps(f, indent=1, allow_nan=False))
    t = dir_out / "tablas"
    texto = informe(s, tablas_descriptivo(s, t), tabla_secundarios(s, t), tabla_estadistica(s, t), tabla_formulas(f, t))
    (dir_out / "informe.md").write_text(texto)
    fig_cajas(grupos, dir_out / "figuras" / "cajas_e_prime.png")
    fig_convergencia(grupos, dir_out / "figuras" / "convergencia.png", ("A", "B", "C_tree", "C_gep"))
    fig_convergencia(grupos, dir_out / "figuras" / "convergencia_arbol_vs_gep.png", ("C_tree", "C_gep"))
    fig_originales(f, dir_out / "figuras" / "originales_conservados.png")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="directorio results/runs")
    ap.add_argument("--out", type=Path, default=RAIZ / "results" / "analysis")
    args = ap.parse_args()
    s = ejecutar(args.runs, args.out)
    print((args.out / "informe.md").read_text())
    print(f"\nEscrito en {args.out}")
    _ = s


if __name__ == "__main__":
    main()
