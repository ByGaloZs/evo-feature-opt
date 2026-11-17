"""Análisis exploratorio de la tasa de mutación de los árboles (posterior a la validación, SOLO train).

Contrasta H6 de ``docs/EXPLORATORIO.md``: ¿explica la baja tasa de mutación de fórmulas de los árboles
(0,2 operaciones por individuo frente a ≈ 2,6 en GEP) las ejecuciones malas de C con árboles en California?

- Solo California, C con árboles, con todo igual que ``configs/final_C_tree.toml`` salvo ``p_mutation``.
- Brazos nuevos: ``p_mutation`` = 0,5 y 1,0, semillas 0 a 29 (60 ejecuciones).
- Control (0,2): se reutiliza el MAE en CV de re-selección de las 30 ejecuciones finales. Antes de lanzar los
  brazos se vuelven a ejecutar las semillas 0, 1 y 2 y se comprueba que ``cv_mae_best`` coincide EXACTAMENTE con
  el de los JSON finales; si no coincide, se para (código de salida 3) y no se ejecuta nada más.
- Referencia: ``cv_mae_best`` de las 30 ejecuciones finales de C con GEP.
- Métrica: MAE en CV de re-selección (folds nuevos), como el piloto. No lee validación ni calcula e′: el ajuste
  usa solo ``data.load_train`` y el transformador. No usa ``experiment.py`` ni ``run_experiments.py``.
- Todo se escribe bajo ``results/explore_mutation/`` (``runs/``, ``control_check/``, ``resumen.{md,json}``).
  Es reanudable. No cambia el sistema, los parámetros por defecto, las configs finales ni ``results/runs``.

Los tests de Fisher y Mann-Whitney son exploratorios y no se corrigen por comparaciones múltiples.

Uso: python scripts/run_explore_mutation.py [--dry-run] [--jobs N] [--summarize]
"""

from __future__ import annotations

import os

# un solo hilo de álgebra lineal por proceso (§15.5); debe fijarse antes de importar NumPy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from evofeat import data
from evofeat.transformer import EvoFeatureTransformer, RunConfig, load_config

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "results" / "explore_mutation"
DATASET = "california"
CONFIG_FINAL = RAIZ / "configs" / "final_C_tree.toml"
FINAL_ARBOL = RAIZ / "results" / "runs" / "final_C_tree" / DATASET
FINAL_GEP = RAIZ / "results" / "runs" / "final_C_gep" / DATASET
BRAZOS = (0.5, 1.0)
P_CONTROL = 0.2
SEMILLAS = tuple(range(30))
SEMILLAS_COMPROBACION = (0, 1, 2)
UMBRAL_CV = 0.5  # el umbral de la sección 5.2


class ControlNoCoincide(RuntimeError):
    """El control (p_mutation = 0,2) no reproduce exactamente el MAE en CV de los JSON finales."""


def _diagnostico():
    """Reutiliza las funciones de la sección 5.2 (misma heurística de transformaciones y de generación)."""
    spec = importlib.util.spec_from_file_location("diagnose_tail_aux", RAIZ / "scripts" / "diagnose_tail.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------------------------
# Configuración y ejecución (solo train)
# ---------------------------------------------------------------------------------------------

def config_brazo(p_mutation: float, overrides: dict[str, Any] | None = None) -> tuple[RunConfig, str]:
    """Configuración final de C con árboles con otro ``p_mutation``; devuelve (cfg para California, hash)."""
    cfg = load_config(str(CONFIG_FINAL))
    cfg = replace(cfg, name=f"explore_pm{p_mutation:g}", p_mutation=p_mutation, **(overrides or {}))
    h = hashlib.sha256(CONFIG_FINAL.read_bytes() + f"|explore|{p_mutation:g}".encode()).hexdigest()
    return cfg.for_dataset(DATASET), h


def ruta_json(base: Path, p_mutation: float, seed: int, control: bool = False) -> Path:
    carpeta = "control_check" if control else f"runs/pm{p_mutation:g}"
    return Path(base) / carpeta / DATASET / f"seed_{seed}.json"


def ejecutar_semilla(p_mutation: float, seed: int, base: Path, overrides: dict[str, Any] | None = None,
                     control: bool = False) -> tuple[str, bool]:
    """Ajusta el transformador con el train de California y guarda el registro. Solo train. Reanudable."""
    ruta = ruta_json(base, p_mutation, seed, control)
    if ruta.exists():
        return str(ruta), False
    cfg, h = config_brazo(p_mutation, overrides)
    tr = data.load_train(DATASET)
    t0 = time.perf_counter()
    tf = EvoFeatureTransformer(cfg, seed).fit(tr.X, tr.y, feature_names=tr.nombres)
    seg = time.perf_counter() - t0
    registro = {
        "meta": {"p_mutation": p_mutation, "seed": seed, "dataset": DATASET, "config_hash": h, "control_check": control,
                 "epsilon": cfg.epsilon},
        "cv_mae_best": tf.cv_mae_best_, "evolution_best_fitness": tf.evolution_best_fitness_,
        "complexity": tf.complexity_, "n_columns": tf.n_columns, "kept_originals": tf.kept_originals_,
        "formulas": tf.formulas_, "individuals_generated": tf.individuals_generated_,
        "distinct_evaluations": tf.distinct_evaluations_, "cache_hits": tf.cache_hits_,
        "seconds_fit": seg, "history": tf.history_,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registro, indent=1, allow_nan=False))
    tmp.replace(ruta)
    return str(ruta), True


def _tarea(args: tuple) -> tuple[float, int, bool, float]:
    p, seed, base, overrides, control = args
    t0 = time.perf_counter()
    _, hecha = ejecutar_semilla(p, seed, Path(base), overrides, control)
    return p, seed, control, time.perf_counter() - t0


def comprobar_control(base: Path = BASE, dir_final: Path = FINAL_ARBOL, semillas: tuple[int, ...] = SEMILLAS_COMPROBACION,
                      overrides: dict[str, Any] | None = None, jobs: int = 1) -> list[dict[str, Any]]:
    """Reejecuta las semillas de control con p_mutation = 0,2 y exige igualdad EXACTA con los JSON finales."""
    _ejecutar_todo([(P_CONTROL, s, str(base), overrides, True) for s in semillas], jobs, verbose=False)
    filas = []
    for s in semillas:
        nuevo = json.loads(ruta_json(base, P_CONTROL, s, True).read_text())["cv_mae_best"]
        final = json.loads((Path(dir_final) / f"seed_{s}.json").read_text())["cv_mae_best"]
        filas.append({"seed": s, "final_cv_mae_best": final, "rerun_cv_mae_best": nuevo, "equal": nuevo == final})
    (Path(base) / "control_check").mkdir(parents=True, exist_ok=True)
    (Path(base) / "control_check" / "comprobacion.json").write_text(json.dumps(filas, indent=1, allow_nan=False))
    if not all(f["equal"] for f in filas):
        raise ControlNoCoincide("; ".join(f"semilla {f['seed']}: final {f['final_cv_mae_best']!r} ≠ reejecución {f['rerun_cv_mae_best']!r}"
                                          for f in filas if not f["equal"]))
    return filas


def _ejecutar_todo(tareas: list[tuple], jobs: int, verbose: bool = True) -> dict[str, Any]:
    res: dict[str, Any] = {"pendientes": len(tareas), "hechas": 0, "fallos": []}
    t0 = time.perf_counter()

    def ok(r: tuple[float, int, bool, float]) -> None:
        res["hechas"] += 1
        if verbose:
            p, seed, control, seg = r
            etiqueta = "control" if control else f"p_mutation {p:g}"
            print(f"[{res['hechas']}/{len(tareas)}] {etiqueta} semilla {seed}: {seg:.0f} s ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    if jobs <= 1:
        for t in tareas:
            try:
                ok(_tarea(t))
            except Exception:  # noqa: BLE001
                res["fallos"].append((t[0], t[1], traceback.format_exc()))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futuros = {pool.submit(_tarea, t): t for t in tareas}
            for f in as_completed(futuros):
                t = futuros[f]
                try:
                    ok(f.result())
                except Exception:  # noqa: BLE001
                    res["fallos"].append((t[0], t[1], traceback.format_exc()))
    return res


def pendientes(base: Path = BASE, brazos: tuple[float, ...] = BRAZOS, semillas: tuple[int, ...] = SEMILLAS,
               overrides: dict[str, Any] | None = None) -> list[tuple]:
    """Ejecuciones de brazos que faltan. Falla si un JSON existente se generó con otra configuración."""
    tareas = []
    for p in brazos:
        _, h = config_brazo(p, overrides)
        for s in semillas:
            ruta = ruta_json(base, p, s)
            if ruta.exists():
                previo = json.loads(ruta.read_text())["meta"].get("config_hash")
                if previo != h:
                    raise RuntimeError(f"{ruta} se generó con otra configuración (hash distinto): no se mezclan resultados")
                continue
            tareas.append((p, s, str(base), overrides, False))
    return tareas


def lanzar(base: Path = BASE, jobs: int = 1, brazos: tuple[float, ...] = BRAZOS, semillas: tuple[int, ...] = SEMILLAS,
           semillas_comprobacion: tuple[int, ...] = SEMILLAS_COMPROBACION, overrides: dict[str, Any] | None = None,
           dir_final: Path = FINAL_ARBOL, verbose: bool = True) -> dict[str, Any]:
    """Comprobación del control y, solo si coincide exactamente, los brazos nuevos."""
    comprobar_control(base, dir_final, semillas_comprobacion, overrides, jobs)
    if verbose:
        print("Control verificado: las semillas", semillas_comprobacion, "reproducen exactamente el MAE en CV final.", flush=True)
    return _ejecutar_todo(pendientes(base, brazos, semillas, overrides), jobs, verbose)


def estimar(jobs: int, base: Path = BASE) -> dict[str, float]:
    """Número de ejecuciones y tiempo estimado a partir de las ejecuciones finales de California."""
    def mediana(carpeta: Path) -> float:
        return float(np.median([json.loads(f.read_text())["seconds_fit"] for f in carpeta.glob("seed_*.json")]))

    arbol, gep = mediana(FINAL_ARBOL), mediana(FINAL_GEP)
    n_brazos = len(pendientes(base))
    n_check = sum(not ruta_json(base, P_CONTROL, s, True).exists() for s in SEMILLAS_COMPROBACION)
    n = n_brazos + n_check
    return {"runs": n, "arm_runs": n_brazos, "control_check_runs": n_check, "jobs": jobs,
            "seconds_per_run_low": arbol, "seconds_per_run_high": gep,
            "wall_minutes_low": n * arbol / max(1, jobs) / 60, "wall_minutes_high": n * gep / max(1, jobs) / 60}


# ---------------------------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------------------------

def _cargar(carpeta: Path) -> list[dict[str, Any]]:
    rs = [json.loads(f.read_text()) for f in sorted(Path(carpeta).glob("seed_*.json"))]
    return sorted(rs, key=lambda r: r["meta"]["seed"])


def _complejidad(r: dict[str, Any]) -> float:
    """Complejidad del ganador: en los JSON finales es la del primer candidato del hall of fame."""
    return float(r["complexity"]) if "complexity" in r else float(r["hall_of_fame"][0]["complexity"])


def resumen_brazo(rs: list[dict[str, Any]]) -> dict[str, Any]:
    dg = _diagnostico()
    cv = np.array([r["cv_mae_best"] for r in rs], dtype=float)
    q1, med, q3 = np.percentile(cv, [25, 50, 75])
    gens = [dg.generacion_bajo(r["history"], UMBRAL_CV) for r in rs]
    alcanzan = [g for g in gens if g is not None]
    return {
        "n": len(rs), "n_cv_below_threshold": int(np.sum(cv < UMBRAL_CV)), "pct_cv_below_threshold": float(100 * np.mean(cv < UMBRAL_CV)),
        "cv_median": float(med), "cv_q1": float(q1), "cv_q3": float(q3), "cv_iqr": float(q3 - q1),
        "pct_transforms_AveOccup": float(100 * np.mean([dg.transforma_ave_occup(r) for r in rs])),
        "pct_log_AveRooms": float(100 * np.mean([dg.log_ave_rooms(r) for r in rs])),
        "complexity_median": float(np.median([_complejidad(r) for r in rs])),
        "distinct_evaluations_median": float(np.median([r["distinct_evaluations"] for r in rs])),
        "seconds_fit_median": float(np.median([r["seconds_fit"] for r in rs])),
        "generation_below_threshold_median": float(np.median(alcanzan)) if alcanzan else None,
        "runs_reaching_threshold": len(alcanzan),
    }


def _tests(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, float]:
    ya, yb = sum(r["cv_mae_best"] < UMBRAL_CV for r in a), sum(r["cv_mae_best"] < UMBRAL_CV for r in b)
    tabla = [[ya, len(a) - ya], [yb, len(b) - yb]]
    fisher = stats.fisher_exact(tabla, alternative="two-sided")
    mw = stats.mannwhitneyu([r["cv_mae_best"] for r in a], [r["cv_mae_best"] for r in b], alternative="two-sided")
    return {"fisher_p": float(fisher.pvalue), "mann_whitney_U": float(mw.statistic), "mann_whitney_p": float(mw.pvalue)}


def resumir(base: Path = BASE, dir_control: Path = FINAL_ARBOL, dir_gep: Path = FINAL_GEP,
            brazos: tuple[float, ...] = BRAZOS) -> dict[str, Any]:
    grupos: dict[str, list[dict[str, Any]]] = {f"pm{P_CONTROL:g} (control)": _cargar(dir_control)}
    for p in brazos:
        grupos[f"pm{p:g}"] = _cargar(Path(base) / "runs" / f"pm{p:g}" / DATASET)
    grupos["GEP (referencia)"] = _cargar(dir_gep)
    grupos = {k: v for k, v in grupos.items() if v}
    nombres = list(grupos)
    control, gep = nombres[0], "GEP (referencia)"
    tests = {}
    for k in nombres:
        if k in (control, gep):
            continue
        tests[k] = {"vs_control": _tests(grupos[k], grupos[control]), **({"vs_gep": _tests(grupos[k], grupos[gep])} if gep in grupos else {})}
    comp_ruta = Path(base) / "control_check" / "comprobacion.json"
    return {"arms": {k: resumen_brazo(v) for k, v in grupos.items()}, "tests_exploratory_uncorrected": tests,
            "control_check": json.loads(comp_ruta.read_text()) if comp_ruta.exists() else None}


def informe(r: dict[str, Any]) -> str:
    p = ["Exploratorio, posterior a la validación y **solo con train** (MAE en CV de re-selección, folds nuevos). "
         "No cambia los resultados finales: la configuración evaluada en validación sigue siendo `p_mutation = 0,2`.", ""]
    if r["control_check"]:
        p += ["**Comprobación del control:** las semillas " + ", ".join(str(f["seed"]) for f in r["control_check"])
              + " reejecutadas con `p_mutation = 0,2` dan exactamente el mismo MAE en CV que los JSON finales: "
              + ("sí" if all(f["equal"] for f in r["control_check"]) else "**NO**") + ".", ""]
    p += ["| Brazo | n | % CV < 0,5 | mediana CV | Q1 | Q3 | RIC | % transforma AveOccup | % log(1 + \\|AveRooms\\|) | complejidad | eval. distintas | tiempo (s) | gen. mediana bajo 0,5 (ejecuciones) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, a in r["arms"].items():
        gen = "—" if a["generation_below_threshold_median"] is None else f"{a['generation_below_threshold_median']:g}"
        p.append(f"| {k} | {a['n']} | {a['pct_cv_below_threshold']:.0f} % ({a['n_cv_below_threshold']}/{a['n']}) | {a['cv_median']:.4f} | "
                 f"{a['cv_q1']:.4f} | {a['cv_q3']:.4f} | {a['cv_iqr']:.4f} | {a['pct_transforms_AveOccup']:.0f} % | {a['pct_log_AveRooms']:.0f} % | "
                 f"{a['complexity_median']:g} | {a['distinct_evaluations_median']:.0f} | {a['seconds_fit_median']:.0f} | {gen} ({a['runs_reaching_threshold']}/{a['n']}) |")
    p += ["", "Tests **exploratorios, sin corregir por comparaciones múltiples** (bilaterales):", "",
          "| Brazo | Frente a | Fisher exacto (proporción CV < 0,5), p | Mann-Whitney (MAE en CV), p |", "| --- | --- | --- | --- |"]
    for k, t in r["tests_exploratory_uncorrected"].items():
        for contra, nombre in (("vs_control", "control (0,2)"), ("vs_gep", "GEP")):
            if contra in t:
                p.append(f"| {k} | {nombre} | {t[contra]['fisher_p']:.3g} | {t[contra]['mann_whitney_p']:.3g} |")
    return "\n".join(p) + "\n"


def escribir_resumen(base: Path = BASE, **kwargs: Any) -> dict[str, Any]:
    r = resumir(base, **kwargs)
    Path(base).mkdir(parents=True, exist_ok=True)
    (Path(base) / "resumen.json").write_text(json.dumps(r, indent=1, allow_nan=False))
    (Path(base) / "resumen.md").write_text(informe(r))
    return r


def _nucleos() -> int:
    spec = importlib.util.spec_from_file_location("run_experiments_aux", RAIZ / "scripts" / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return max(1, mod.nucleos_fisicos() - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="solo número de ejecuciones y tiempo estimado")
    ap.add_argument("--summarize", action="store_true", help="solo regenera el resumen con lo que ya hay")
    args = ap.parse_args()
    jobs = args.jobs or _nucleos()
    if args.summarize:
        print(informe(escribir_resumen()))
        return 0
    try:
        est = estimar(jobs)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Ejecuciones pendientes: {est['runs']} ({est['arm_runs']} de los brazos p_mutation = 0,5 y 1,0 con semillas 0–29 "
          f"+ {est['control_check_runs']} de comprobación del control)")
    print(f"Tiempo estimado con {jobs} procesos: {est['wall_minutes_low']:.0f}–{est['wall_minutes_high']:.0f} min de reloj "
          f"({est['seconds_per_run_low']:.0f}–{est['seconds_per_run_high']:.0f} s por ejecución: la de C con árboles y la de C con GEP finales)")
    if args.dry_run:
        return 0
    try:
        r = lanzar(BASE, jobs)
    except ControlNoCoincide as e:
        print(f"PARO: el control no coincide con los JSON finales: {e}", file=sys.stderr)
        return 3
    print(f"Hechas ahora: {r['hechas']}; fallos: {len(r['fallos'])}")
    for p, seed, tb in r["fallos"]:
        print(f"FALLO p_mutation {p} semilla {seed}:\n{tb}", file=sys.stderr)
    print(informe(escribir_resumen()))
    return 1 if r["fallos"] else 0


if __name__ == "__main__":
    sys.exit(main())
