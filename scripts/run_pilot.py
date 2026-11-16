"""Piloto de D20: rejilla K × profundidad máxima con la configuración C (árboles).

Implementa D20, D17 (ε) y §15.6 (hito 8). Rejilla K ∈ {3, 5, 7} × profundidad ∈ {2, 3, 4}
(9 combinaciones) con las semillas 100, 101 y 102, usando SOLO train: no importa ni llama a
nada de validación (RNF-02). La métrica es el MAE del ganador en la CV repetida con folds
nuevos (``cv_mae_best``, D7), que no está sesgado por la selección.

Salida:
- ``results/pilot/<dataset>/K<k>_d<d>_seed<s>.json``: un registro por ejecución (el script
  salta las que ya existen).
- ``results/pilot/<dataset>/summary.json`` y una tabla en pantalla con MAE en CV,
  complejidad y tiempos, más la PROPUESTA de K, profundidad y ε.

El script no decide nada: tú fijas los valores (hito 8 → parada) y después se escribe
``docs/HIPOTESIS.md`` antes del hito 9.

Propuesta de ε (D17): por ejecución, desviación típica muestral de los MAE de los 5 folds
del ganador dividida por √5; se propone la mediana de las 27 ejecuciones. Propuesta de
combinación (D20): la de menor MAE medio en CV; si varias quedan dentro de ε de ella, la de
menor K y, a igualdad, menor profundidad.

Uso: python scripts/run_pilot.py --dataset california
"""

from __future__ import annotations

import os

# un solo hilo de álgebra lineal por proceso (§15.5); debe fijarse antes de importar NumPy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from evofeat import data
from evofeat.transformer import EvoFeatureTransformer, RunConfig

RAIZ = Path(__file__).resolve().parents[1]
DIR_PILOTO = RAIZ / "results" / "pilot"

REJILLA_K = (3, 5, 7)
REJILLA_PROFUNDIDAD = (2, 3, 4)
SEMILLAS = (100, 101, 102)  # §15.5: distintas de las finales


def _ruta(directorio: Path, dataset: str, K: int, depth: int, seed: int) -> Path:
    return directorio / dataset / f"K{K}_d{depth}_seed{seed}.json"


def run_cell(args: tuple[str, int, int, int, str, dict[str, Any]]) -> tuple[str, bool]:
    """Una ejecución del piloto. Devuelve (ruta, ejecutada). Solo lee train."""
    dataset, K, depth, seed, directorio, overrides = args
    ruta = _ruta(Path(directorio), dataset, K, depth, seed)
    if ruta.exists():
        return str(ruta), False
    tr = data.load_train(dataset)
    cfg = RunConfig(
        name="pilot", configuration="C", representation="tree", K=K, max_depth=depth,
        epsilon=0.0, **overrides,
    )
    t0 = time.perf_counter()
    tf = EvoFeatureTransformer(cfg, seed).fit(tr.X, tr.y, feature_names=tr.nombres)
    segundos = time.perf_counter() - t0
    maes = tf.cv_fold_maes_
    registro = {
        "dataset": dataset, "K": K, "max_depth": depth, "seed": seed,
        "cv_mae_best": tf.cv_mae_best_,  # CV repetida con folds nuevos (D7)
        "evolution_best_fitness": tf.evolution_best_fitness_,
        "cv_fold_maes": maes,
        "epsilon_run": float(np.std(maes, ddof=1) / math.sqrt(len(maes))),
        "complexity": tf.complexity_,
        "n_columns": tf.n_columns,
        "kept_originals": tf.kept_originals_,
        "formulas": tf.formulas_,
        "individuals_generated": tf.individuals_generated_,
        "distinct_evaluations": tf.distinct_evaluations_,
        "cache_hits": tf.cache_hits_,
        "seconds_fit": segundos,
        "budget": {"pop_size": cfg.pop_size, "generations": cfg.generations, "hof_size": cfg.hof_size,
                   "repeats": cfg.repeats},
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registro, indent=1, allow_nan=False))
    tmp.replace(ruta)
    return str(ruta), True


def resumir(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Tabla por combinación, ε propuesto y combinación propuesta (D17, D20)."""
    combos: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in registros:
        combos.setdefault((r["K"], r["max_depth"]), []).append(r)
    filas = []
    for (K, d), rs in sorted(combos.items()):
        cv = [r["cv_mae_best"] for r in rs]
        filas.append({
            "K": K, "max_depth": d, "runs": len(rs),
            "cv_mae_mean": float(np.mean(cv)),
            "cv_mae_sd": float(np.std(cv, ddof=1)) if len(cv) > 1 else 0.0,
            "complexity_mean": float(np.mean([r["complexity"] for r in rs])),
            "n_columns_mean": float(np.mean([r["n_columns"] for r in rs])),
            "seconds_mean": float(np.mean([r["seconds_fit"] for r in rs])),
            "distinct_evaluations_mean": float(np.mean([r["distinct_evaluations"] for r in rs])),
        })
    epsilon = float(np.median([r["epsilon_run"] for r in registros]))
    mejor = min(filas, key=lambda f: f["cv_mae_mean"])
    dentro = [f for f in filas if f["cv_mae_mean"] - mejor["cv_mae_mean"] < epsilon]
    elegido = min(dentro, key=lambda f: (f["K"], f["max_depth"]))
    return {
        "rows": filas,
        "epsilon_proposed": epsilon,
        "epsilon_run_range": [min(r["epsilon_run"] for r in registros), max(r["epsilon_run"] for r in registros)],
        "best_raw": {"K": mejor["K"], "max_depth": mejor["max_depth"]},
        "within_epsilon": [{"K": f["K"], "max_depth": f["max_depth"]} for f in dentro],
        "proposed": {"K": elegido["K"], "max_depth": elegido["max_depth"]},
    }


def tabla(resumen: dict[str, Any], dataset: str) -> str:
    """Tabla en markdown con los resultados del piloto."""
    lineas = [
        f"Piloto D20 — {dataset} (configuración C, árboles; MAE en CV repetida con folds nuevos)",
        "",
        "| K | prof. | MAE CV (media ± sd) | complejidad | columnas | tiempo (s) | eval. distintas |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in resumen["rows"]:
        lineas.append(
            f"| {f['K']} | {f['max_depth']} | {f['cv_mae_mean']:.4f} ± {f['cv_mae_sd']:.4f} | "
            f"{f['complexity_mean']:.1f} | {f['n_columns_mean']:.1f} | {f['seconds_mean']:.0f} | "
            f"{f['distinct_evaluations_mean']:.0f} |"
        )
    p, b = resumen["proposed"], resumen["best_raw"]
    lo, hi = resumen["epsilon_run_range"]
    lineas += [
        "",
        f"ε propuesto (mediana de std de los 5 folds / √5): {resumen['epsilon_proposed']:.4f} "
        f"(rango por ejecución {lo:.4f} – {hi:.4f})",
        f"Mejor MAE medio en CV: K = {b['K']}, profundidad = {b['max_depth']}",
        "Dentro de ε del mejor: " + ", ".join(f"(K={c['K']}, d={c['max_depth']})" for c in resumen["within_epsilon"]),
        f"Propuesta (menor K y, a igualdad, menor profundidad): K = {p['K']}, profundidad = {p['max_depth']}",
    ]
    return "\n".join(lineas)


def run_pilot(dataset: str, directorio: Path = DIR_PILOTO, jobs: int = 1,
              seeds: tuple[int, ...] = SEMILLAS, **overrides: Any) -> dict[str, Any]:
    """Ejecuta la rejilla (saltando lo ya hecho), escribe el resumen y lo devuelve."""
    tareas = [
        (dataset, K, d, s, str(directorio), overrides)
        for K in REJILLA_K for d in REJILLA_PROFUNDIDAD for s in seeds
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            resultados = list(pool.map(run_cell, tareas))
    else:
        resultados = [run_cell(t) for t in tareas]
    registros = [json.loads(Path(r).read_text()) for r, _ in resultados]
    resumen = resumir(registros)
    resumen["executed_now"] = sum(1 for _, ejecutada in resultados if ejecutada)
    resumen["dataset"] = dataset
    (directorio / dataset).mkdir(parents=True, exist_ok=True)
    (directorio / dataset / "summary.json").write_text(json.dumps(resumen, indent=1, allow_nan=False))
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(data.DATASETS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="procesos en paralelo (por defecto núcleos − 1)")
    ap.add_argument("--out-dir", type=Path, default=DIR_PILOTO)
    args = ap.parse_args()
    t0 = time.perf_counter()
    resumen = run_pilot(args.dataset, args.out_dir, args.jobs)
    print(tabla(resumen, args.dataset))
    print(f"\nEjecuciones nuevas: {resumen['executed_now']}; tiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
