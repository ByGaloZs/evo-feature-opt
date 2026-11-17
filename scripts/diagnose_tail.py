"""Diagnóstico de la cola de ejecuciones malas de C con árboles en California (posterior a la validación).

Solo diagnóstico: lee los JSON de ``results/runs`` y escribe ``results/diagnostics/california_c_tree_tail.md``
y ``.json``. No cambia ningún parámetro ni el código del sistema; cualquier cambio sería una decisión
posterior a la validación y la toma el usuario.

Preguntas: ¿las ejecuciones malas tienen también un MAE en CV malo (fallo de búsqueda) o buen CV y mal e′
(no generalizan)? ¿les faltan transformaciones que sí tienen las buenas? ¿son demasiado simples? ¿había en
el hall of fame un individuo mejor en CV que no se eligió?

Uso: python scripts/diagnose_tail.py results/runs [--out results/diagnostics]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
DATASET = "california"
UMBRAL_BRECHA = 0.02  # e′ − MAE en CV por encima del cual una ejecución «no generaliza» (diagnóstico)
UMBRAL_CV_GENERACION = 0.50  # MAE en CV bajo el cual se considera que la búsqueda ya despegó (diagnóstico)


def cargar(dir_runs: Path, experimento: str, dataset: str = DATASET) -> list[dict[str, Any]]:
    rs = [json.loads(f.read_text()) for f in sorted((Path(dir_runs) / experimento / dataset).glob("seed_*.json"))]
    return sorted(rs, key=lambda r: r["meta"]["seed"])


def ultima_mejora(historial: list[dict[str, Any]]) -> int:
    """Última generación en la que el mejor fitness en CV mejoró."""
    mejor = [h["best_fitness"] for h in historial]
    ultima = 0
    for g in range(1, len(mejor)):
        if mejor[g] < mejor[g - 1] - 1e-15:
            ultima = g
    return ultima


def generacion_bajo(historial: list[dict[str, Any]], umbral: float) -> int | None:
    for g, h in enumerate(historial):
        if h["best_fitness"] <= umbral:
            return g
    return None


def transforma_ave_occup(r: dict[str, Any]) -> bool:
    """Alguna fórmula combina AveOccup con sqrt, log o un cociente (transformación compresora)."""
    return any("AveOccup" in f and ("sqrt(" in f or "log(" in f or " / " in f) for f in r["formulas"])


def log_ave_rooms(r: dict[str, Any]) -> bool:
    return any("log(1 + |AveRooms|)" in f for f in r["formulas"])


def ficha(r: dict[str, Any]) -> dict[str, Any]:
    hof = r["hall_of_fame"]
    ganador = hof[0]  # la re-selección ordena por MAE en folds nuevos, luego por complejidad
    return {
        "seed": r["meta"]["seed"], "e_prime": r["mae"], "cv_mae_best": r["cv_mae_best"],
        "cv_mae_evolution": r["evolution_best_fitness"], "gap_e_prime_minus_cv": r["mae"] - r["cv_mae_best"],
        "n_columns": r["n_columns"], "kept_originals": r["kept_originals"], "formulas": r["formulas"],
        "complexity": ganador["complexity"], "last_improvement_generation": ultima_mejora(r["history"]),
        "generation_below_threshold": generacion_bajo(r["history"], UMBRAL_CV_GENERACION),
        "hof_min_cv_evolution": min(h["cv_mae_evolution"] for h in hof),
        "hof_winner_cv_evolution": ganador["cv_mae_evolution"],
        "hof_min_cv_new_folds": min(h["cv_mae_new_folds"] for h in hof),
        "hof_winner_cv_new_folds": ganador["cv_mae_new_folds"],
        "hof_size": len(hof),
        "transforms_AveOccup": transforma_ave_occup(r), "log_AveRooms": log_ave_rooms(r),
    }


def corte_por_mayor_hueco(valores: np.ndarray) -> float:
    """Umbral entre ejecuciones buenas y malas: el mayor hueco entre e′ consecutivos por encima de la mediana,
    sin contar el valor más alto (un atípico aislado ocultaría la separación del grupo)."""
    v = np.sort(valores)
    inicio = len(v) // 2
    huecos = np.diff(v[inicio:-1])
    i = int(np.argmax(huecos)) + inicio
    return float((v[i] + v[i + 1]) / 2)


def resumen_grupo(fichas: list[dict[str, Any]]) -> dict[str, float]:
    def med(campo: str) -> float:
        return float(np.median([f[campo] for f in fichas]))

    def frac(campo: str) -> float:
        return float(np.mean([bool(f[campo]) for f in fichas]))

    return {
        "n": len(fichas), "e_prime_median": med("e_prime"), "cv_mae_best_median": med("cv_mae_best"),
        "gap_median": med("gap_e_prime_minus_cv"), "complexity_median": med("complexity"),
        "n_columns_median": med("n_columns"), "last_improvement_median": med("last_improvement_generation"),
        "still_improving_at_end_fraction": float(np.mean([f["last_improvement_generation"] >= 95 for f in fichas])),
        "transforms_AveOccup_fraction": frac("transforms_AveOccup"), "log_AveRooms_fraction": frac("log_AveRooms"),
        "MedInc_kept_fraction": float(np.mean(["MedInc" in f["kept_originals"] for f in fichas])),
    }


def eventos_de_mutacion(cfg_tree: dict[str, Any], cfg_gep: dict[str, Any]) -> dict[str, float]:
    """Número esperado de operaciones de mutación sobre fórmulas por individuo (aritmética de los TOML)."""
    longitud_gen = 2 * cfg_gep["head"] + 1
    por_gen = cfg_gep["p_point"] * longitud_gen + cfg_gep["p_is"] + cfg_gep["p_ris"]
    return {"tree_per_individual": cfg_tree["p_mutation"], "gep_per_individual": cfg_gep["K"] * por_gen,
            "gep_gene_length": longitud_gen}


def md_ficha(f: dict[str, Any]) -> str:
    formulas = "; ".join(f"`{x}`" for x in f["formulas"])
    return (f"| {f['seed']} | {f['e_prime']:.4f} | {f['cv_mae_best']:.4f} | {f['gap_e_prime_minus_cv']:+.4f} | {f['n_columns']} | "
            f"{', '.join(f['kept_originals'])} | {formulas} | {f['complexity']} | {f['last_improvement_generation']} |")


CAB = ("| semilla | e′ | MAE CV | e′ − CV | columnas | originales conservados | fórmulas | complejidad | última mejora (gen.) |\n"
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def diagnosticar(dir_runs: Path, cfgs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    arbol, gep = cargar(dir_runs, "final_C_tree"), cargar(dir_runs, "final_C_gep")
    fa, fg = [ficha(r) for r in arbol], [ficha(r) for r in gep]
    ep = np.array([f["e_prime"] for f in fa])
    corte = corte_por_mayor_hueco(ep)
    malas, buenas = [f for f in fa if f["e_prime"] > corte], [f for f in fa if f["e_prime"] <= corte]
    orden = sorted(fa, key=lambda f: -f["e_prime"])
    out: dict[str, Any] = {
        "cut_e_prime": corte, "worst_tree": orden[:5], "best_tree": orden[-5:][::-1],
        "worst_gep": sorted(fg, key=lambda f: -f["e_prime"])[:3],
        "groups": {"bad": resumen_grupo(malas), "good": resumen_grupo(buenas)},
        "spearman_e_prime_vs_cv": {
            "tree": stats.spearmanr(ep, [f["cv_mae_best"] for f in fa])._asdict() | {},
            "gep": stats.spearmanr([f["e_prime"] for f in fg], [f["cv_mae_best"] for f in fg])._asdict() | {}},
        "hof_alternatives_bad_runs": [{"seed": f["seed"], "winner_cv_evolution": f["hof_winner_cv_evolution"],
                                       "hof_min_cv_evolution": f["hof_min_cv_evolution"],
                                       "winner_cv_new_folds": f["hof_winner_cv_new_folds"],
                                       "hof_min_cv_new_folds": f["hof_min_cv_new_folds"]} for f in malas],
        "convergence": {}, "ratio_bad_to_total": len(malas) / len(fa),
        "large_gap_runs": {"threshold": UMBRAL_BRECHA,
                           "tree": [{"seed": f["seed"], "e_prime": f["e_prime"], "cv": f["cv_mae_best"], "gap": f["gap_e_prime_minus_cv"], "formulas": f["formulas"]}
                                    for f in fa if f["gap_e_prime_minus_cv"] > UMBRAL_BRECHA],
                           "gep": [{"seed": f["seed"], "e_prime": f["e_prime"], "cv": f["cv_mae_best"], "gap": f["gap_e_prime_minus_cv"], "formulas": f["formulas"]}
                                   for f in fg if f["gap_e_prime_minus_cv"] > UMBRAL_BRECHA]},
    }
    for nombre, fs in (("tree", fa), ("gep", fg)):
        gens = [f["generation_below_threshold"] for f in fs]
        alcanzan = [g for g in gens if g is not None]
        out["convergence"][nombre] = {"threshold": UMBRAL_CV_GENERACION, "runs_reaching": len(alcanzan), "runs": len(fs),
                                      "median_generation_when_reached": float(np.median(alcanzan)) if alcanzan else None,
                                      "e_prime_iqr": float(np.subtract(*np.percentile([f["e_prime"] for f in fs], [75, 25])))}
    if cfgs:
        out["mutation_events"] = eventos_de_mutacion(cfgs["tree"], cfgs["gep"])
    return out


def _spearman(d: dict[str, Any]) -> str:
    return f"ρ = {d['statistic']:.3f} (p = {d['pvalue']:.2g})"


def informe(d: dict[str, Any]) -> str:
    b, g = d["groups"]["bad"], d["groups"]["good"]
    cv = d["convergence"]
    p = ["# Diagnóstico de la cola de C con árboles en California (posterior a la validación)", "",
         "Generado por `scripts/diagnose_tail.py`. Solo diagnóstico: no se ha cambiado ningún parámetro ni el código del sistema.", "",
         "## Las 5 peores y las 5 mejores ejecuciones de C con árboles, y las 3 peores de C con GEP", "",
         "**Peores, C con árboles**", "", CAB] + [md_ficha(f) for f in d["worst_tree"]] + [
         "", "**Mejores, C con árboles**", "", CAB] + [md_ficha(f) for f in d["best_tree"]] + [
         "", "**Peores, C con GEP (comparación)**", "", CAB] + [md_ficha(f) for f in d["worst_gep"]] + [
         "", "La complejidad es la del individuo ganador (columnas + nodos). «Última mejora» es la última generación en que "
             "mejoró el mejor fitness de la evolución (100 = seguía mejorando al terminar).", "",
         "## ¿Mal CV (fallo de búsqueda) o buen CV y mal e′ (no generaliza)?", "",
         f"- Correlación de Spearman entre e′ y el MAE en CV del ganador: árboles {_spearman(d['spearman_e_prime_vs_cv']['tree'])}; "
         f"GEP {_spearman(d['spearman_e_prime_vs_cv']['gep'])}.",
         f"- Las ejecuciones se separan en dos grupos por el mayor hueco en e′ (corte {d['cut_e_prime']:.4f}): "
         f"{b['n']} malas y {g['n']} buenas ({100 * d['ratio_bad_to_total']:.0f} % malas).",
         f"- MAE en CV mediano: malas {b['cv_mae_best_median']:.4f} frente a buenas {g['cv_mae_best_median']:.4f}; "
         f"e′ mediano: malas {b['e_prime_median']:.4f} frente a buenas {g['e_prime_median']:.4f}; "
         f"e′ − CV mediano: malas {b['gap_median']:+.4f}, buenas {g['gap_median']:+.4f}.",
         "- Es decir, el e′ malo ya se ve en la CV sobre train: es un fallo de búsqueda (la evolución no encontró las transformaciones útiles), no de generalización.",
         "- Excepciones, con e′ − CV > " + f"{d['large_gap_runs']['threshold']}" + " (buen CV relativo y mal e′): árboles "
         + (", ".join(f"semilla {x['seed']} (e′ = {x['e_prime']:.4f}, CV = {x['cv']:.4f})" for x in d["large_gap_runs"]["tree"]) or "ninguna")
         + "; GEP " + (", ".join(f"semilla {x['seed']} (e′ = {x['e_prime']:.4f}, CV = {x['cv']:.4f})" for x in d["large_gap_runs"]["gep"]) or "ninguna")
         + ". " + ("En todas ellas hay una fórmula con un producto que incluye AveOccup (un término sin acotar); " if (d["large_gap_runs"]["tree"] + d["large_gap_runs"]["gep"]) and all(any("AveOccup" in f and " * " in f for f in x["formulas"]) for x in d["large_gap_runs"]["tree"] + d["large_gap_runs"]["gep"]) else "")
         + "como causa es una hipótesis no verificada: comprobarla exigiría leer la validación.", "",
         "## ¿Qué distingue a las malas de las buenas?", "",
         "| medida | malas | buenas |", "| --- | --- | --- |",
         f"| alguna fórmula transforma AveOccup (sqrt, log o cociente) | {100 * b['transforms_AveOccup_fraction']:.0f} % | {100 * g['transforms_AveOccup_fraction']:.0f} % |",
         f"| alguna fórmula contiene log(1 + \\|AveRooms\\|) | {100 * b['log_AveRooms_fraction']:.0f} % | {100 * g['log_AveRooms_fraction']:.0f} % |",
         f"| conservan MedInc como original | {100 * b['MedInc_kept_fraction']:.0f} % | {100 * g['MedInc_kept_fraction']:.0f} % |",
         f"| complejidad mediana | {b['complexity_median']:g} | {g['complexity_median']:g} |",
         f"| columnas medianas | {b['n_columns_median']:g} | {g['n_columns_median']:g} |",
         f"| última generación con mejora (mediana) | {b['last_improvement_median']:g} | {g['last_improvement_median']:g} |",
         f"| seguían mejorando al terminar (≥ gen. 95) | {100 * b['still_improving_at_end_fraction']:.0f} % | {100 * g['still_improving_at_end_fraction']:.0f} % |",
         "", "La heurística de «transforma AveOccup» mira si una misma fórmula contiene AveOccup y sqrt, log o un cociente; "
             "las fórmulas completas están en las tablas de arriba y en los JSON.", "",
         "## Hall of fame de las ejecuciones malas", "",
         "| semilla | CV evolución del ganador | mejor CV evolución del hall of fame | CV folds nuevos del ganador | mejor CV folds nuevos del hall of fame |",
         "| --- | --- | --- | --- | --- |"]
    p += [f"| {h['seed']} | {h['winner_cv_evolution']:.4f} | {h['hof_min_cv_evolution']:.4f} | {h['winner_cv_new_folds']:.4f} | {h['hof_min_cv_new_folds']:.4f} |"
          for h in d["hof_alternatives_bad_runs"]]
    dif = max(h["winner_cv_evolution"] - h["hof_min_cv_evolution"] for h in d["hof_alternatives_bad_runs"])
    p += ["", f"Ningún alternativo del hall of fame es sensiblemente mejor en CV: la mayor ventaja de un alternativo en la CV de la evolución "
              f"es {dif:.4f}. (El JSON guarda las métricas de los alternativos, no sus fórmulas; su e′ no se ha calculado para no seleccionar con validación.)", "",
          "## Velocidad de la búsqueda", "",
          f"- Ejecuciones cuyo mejor MAE en CV baja de {cv['tree']['threshold']}: árboles {cv['tree']['runs_reaching']}/{cv['tree']['runs']} "
          f"(generación mediana {cv['tree']['median_generation_when_reached']}), GEP {cv['gep']['runs_reaching']}/{cv['gep']['runs']} "
          f"(generación mediana {cv['gep']['median_generation_when_reached']}).",
          f"- RIC de e′: árboles {cv['tree']['e_prime_iqr']:.4f}, GEP {cv['gep']['e_prime_iqr']:.4f}."]
    if "mutation_events" in d:
        m = d["mutation_events"]
        p += ["", "## Una asimetría entre las dos representaciones (hipótesis de causa, no probada)", "",
              f"Con las tasas de los TOML, el número esperado de operaciones de mutación sobre fórmulas por individuo es "
              f"{m['tree_per_individual']:.2f} en árboles (`p_mutation` = 0,2: se muta un árbol en el 20 % de los individuos) frente a "
              f"{m['gep_per_individual']:.2f} en GEP (K genes de {m['gep_gene_length']} símbolos con mutación puntual 0,044 por símbolo, más IS y RIS), "
              f"es decir, unas {m['gep_per_individual'] / m['tree_per_individual']:.0f} veces más. Es una consecuencia de usar «cada representación con sus tasas» (hito 4/6)."]
    return "\n".join(p) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=RAIZ / "results" / "diagnostics")
    args = ap.parse_args()
    from evofeat.transformer import load_config

    cfgs = {}
    for clave, nombre in (("tree", "final_C_tree"), ("gep", "final_C_gep")):
        c = load_config(str(RAIZ / "configs" / f"{nombre}.toml"))
        cfgs[clave] = {"p_mutation": c.p_mutation, "p_point": c.p_point, "p_is": c.p_is, "p_ris": c.p_ris, "K": c.K, "head": c.head}
    d = diagnosticar(args.runs, cfgs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "california_c_tree_tail.json").write_text(json.dumps(d, indent=1, allow_nan=False, default=float))
    texto = informe(d)
    (args.out / "california_c_tree_tail.md").write_text(texto)
    print(texto)


if __name__ == "__main__":
    main()
