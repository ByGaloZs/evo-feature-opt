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

Además del MAE de cada combinación, el resumen incluye el MAE en CV del baseline (OLS con
todos los originales, con los mismos folds repetidos de cada semilla), para saber ya, solo con
train, si C mejora al baseline en CV.

Regla común (decidida tras ver el piloto, sustituye a la de D20): ``--common`` elige la
combinación más pequeña que queda dentro de ε de la mejor en TODOS los datasets.

Hito 8b (``--eps-check``): ejecuta C con K = 3, profundidad 3 y el ε final con las semillas del
piloto y lo compara con ε = 0 (las ejecuciones con ε = 0 son las del propio piloto, que son
deterministas). Si el MAE en CV con ε > 0 empeora más de ε respecto a ε = 0, avisa.

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


def _ruta(directorio: Path, dataset: str, K: int, depth: int, seed: int, etiqueta: str = "") -> Path:
    return directorio / dataset / f"K{K}_d{depth}{etiqueta}_seed{seed}.json"


def nucleos_fisicos() -> int:
    """Núcleos físicos (sin hyperthreading) leídos de /proc/cpuinfo; si no se puede, los lógicos / 2."""
    try:
        pares = set()
        fisico = nucleo = None
        for linea in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not linea.strip():
                if fisico is not None and nucleo is not None:
                    pares.add((fisico, nucleo))
                fisico = nucleo = None
            elif linea.startswith("physical id"):
                fisico = linea.split(":")[1].strip()
            elif linea.startswith("core id"):
                nucleo = linea.split(":")[1].strip()
        if pares:
            return len(pares)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def run_cell(args: tuple[str, int, int, int, str, dict[str, Any]] | tuple[str, int, int, int, str, dict[str, Any], float, str]) -> tuple[str, bool]:
    """Una ejecución del piloto. Devuelve (ruta, ejecutada). Solo lee train.

    Con 8 elementos, los dos últimos son ε y la etiqueta del archivo (comprobación 8b).
    """
    dataset, K, depth, seed, directorio, overrides, *resto = args
    epsilon, etiqueta = resto if resto else (0.0, "")
    ruta = _ruta(Path(directorio), dataset, K, depth, seed, etiqueta)
    if ruta.exists():
        return str(ruta), False
    tr = data.load_train(dataset)
    cfg = RunConfig(
        name="pilot", configuration="C", representation="tree", K=K, max_depth=depth,
        epsilon=epsilon, **overrides,
    )
    t0 = time.perf_counter()
    tf = EvoFeatureTransformer(cfg, seed).fit(tr.X, tr.y, feature_names=tr.nombres)
    segundos = time.perf_counter() - t0
    maes = tf.cv_fold_maes_
    registro = {
        "dataset": dataset, "K": K, "max_depth": depth, "seed": seed, "epsilon": epsilon,
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


def baseline_cv(dataset: str, seeds: tuple[int, ...] = SEMILLAS) -> dict[str, Any]:
    """MAE en CV del baseline: OLS con todos los originales y los mismos folds repetidos (D7).

    Usa la configuración ``baseline`` del transformador, así que los folds son exactamente los
    de la re-selección de cada semilla (``random_state = semilla + 10 000 + r``). Solo train.
    """
    tr = data.load_train(dataset)
    valores = []
    for seed in seeds:
        cfg = RunConfig(name="pilot", configuration="baseline")
        valores.append(EvoFeatureTransformer(cfg, seed).fit(tr.X, tr.y, feature_names=tr.nombres).cv_mae_best_)
    return {"per_seed": valores, "mean": float(np.mean(valores))}


def resumir(registros: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
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
    resumen: dict[str, Any] = {
        "rows": filas,
        "epsilon_proposed": epsilon,
        "epsilon_run_range": [min(r["epsilon_run"] for r in registros), max(r["epsilon_run"] for r in registros)],
        "best_raw": {"K": mejor["K"], "max_depth": mejor["max_depth"]},
        "within_epsilon": [{"K": f["K"], "max_depth": f["max_depth"]} for f in dentro],
        "proposed": {"K": elegido["K"], "max_depth": elegido["max_depth"]},
    }
    if baseline is not None:
        resumen["baseline_cv"] = baseline
        resumen["best_vs_baseline_relative"] = (baseline["mean"] - mejor["cv_mae_mean"]) / baseline["mean"]
        for f in filas:
            f["vs_baseline_relative"] = (baseline["mean"] - f["cv_mae_mean"]) / baseline["mean"]
    return resumen


def proponer_comun(resumenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combinación más pequeña dentro de ε de la mejor en TODOS los datasets (regla común).

    ``resumenes``: dataset -> resumen del piloto, con ε = ``epsilon_proposed``. Desempate:
    menor K y, a igualdad, menor profundidad.
    """
    validas: set[tuple[int, int]] | None = None
    for r in resumenes.values():
        dentro = {(c["K"], c["max_depth"]) for c in r["within_epsilon"]}
        validas = dentro if validas is None else validas & dentro
    if not validas:
        return {"proposed": None, "valid": []}
    K, d = min(validas, key=lambda c: (c[0], c[1]))
    return {"proposed": {"K": K, "max_depth": d}, "valid": [{"K": k, "max_depth": p} for k, p in sorted(validas)]}


def tabla(resumen: dict[str, Any], dataset: str) -> str:
    """Tabla en markdown con los resultados del piloto."""
    con_base = "baseline_cv" in resumen
    lineas = [
        f"Piloto D20 — {dataset} (configuración C, árboles; MAE en CV repetida con folds nuevos)",
        "",
        "| K | prof. | MAE CV (media ± sd) | " + ("vs baseline | " if con_base else "") + "complejidad | columnas | tiempo (s) | eval. distintas |",
        "| --- | --- | --- | " + ("--- | " if con_base else "") + "--- | --- | --- | --- |",
    ]
    for f in resumen["rows"]:
        rel = f"{100 * f['vs_baseline_relative']:+.1f} % | " if con_base else ""
        lineas.append(
            f"| {f['K']} | {f['max_depth']} | {f['cv_mae_mean']:.4f} ± {f['cv_mae_sd']:.4f} | {rel}"
            f"{f['complexity_mean']:.1f} | {f['n_columns_mean']:.1f} | {f['seconds_mean']:.0f} | "
            f"{f['distinct_evaluations_mean']:.0f} |"
        )
    p, b = resumen["proposed"], resumen["best_raw"]
    lo, hi = resumen["epsilon_run_range"]
    if con_base:
        bl = resumen["baseline_cv"]
        lineas += ["", f"Baseline (OLS, todos los originales) en CV con los mismos folds: {bl['mean']:.4f} "
                       f"(por semilla: {', '.join(f'{v:.4f}' for v in bl['per_seed'])}); "
                       f"el mejor C mejora un {100 * resumen['best_vs_baseline_relative']:.1f} %"]
    lineas += [
        "",
        f"ε propuesto (mediana de std de los 5 folds / √5): {resumen['epsilon_proposed']:.4f} "
        f"(rango por ejecución {lo:.4f} – {hi:.4f})",
        f"Mejor MAE medio en CV: K = {b['K']}, profundidad = {b['max_depth']}",
        "Dentro de ε del mejor: " + ", ".join(f"(K={c['K']}, d={c['max_depth']})" for c in resumen["within_epsilon"]),
        f"Propuesta de este dataset (menor K y, a igualdad, menor profundidad): K = {p['K']}, profundidad = {p['max_depth']}",
    ]
    return "\n".join(lineas)


def _tareas(dataset: str, directorio: Path, seeds: tuple[int, ...], overrides: dict[str, Any]) -> list[tuple]:
    return [(dataset, K, d, s, str(directorio), overrides) for K in REJILLA_K for d in REJILLA_PROFUNDIDAD for s in seeds]


def _ejecutar(tareas: list[tuple], jobs: int) -> list[tuple[str, bool]]:
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(run_cell, tareas))
    return [run_cell(t) for t in tareas]


def run_pilot(dataset: str, directorio: Path = DIR_PILOTO, jobs: int = 1,
              seeds: tuple[int, ...] = SEMILLAS, **overrides: Any) -> dict[str, Any]:
    """Ejecuta la rejilla (saltando lo ya hecho), escribe el resumen y lo devuelve."""
    resultados = _ejecutar(_tareas(dataset, directorio, seeds, overrides), jobs)
    registros = [json.loads(Path(r).read_text()) for r, _ in resultados]
    resumen = resumir(registros, baseline_cv(dataset, seeds))
    resumen["executed_now"] = sum(1 for _, ejecutada in resultados if ejecutada)
    resumen["dataset"] = dataset
    (directorio / dataset).mkdir(parents=True, exist_ok=True)
    (directorio / dataset / "summary.json").write_text(json.dumps(resumen, indent=1, allow_nan=False))
    return resumen


def comprobar_epsilon(dataset: str, epsilon: float, directorio: Path = DIR_PILOTO, jobs: int = 1,
                      seeds: tuple[int, ...] = SEMILLAS, K: int = 3, depth: int = 3,
                      **overrides: Any) -> dict[str, Any]:
    """Hito 8b: compara C (K, profundidad) con ε final frente a ε = 0, solo con train.

    Las ejecuciones con ε = 0 son las del piloto (``K<k>_d<d>_seed<s>.json``, deterministas); se
    ejecutan si faltan. El criterio de parada: si el MAE en CV medio con ε > 0 empeora más de ε
    respecto a ε = 0.
    """
    tareas0 = [(dataset, K, depth, s, str(directorio), overrides) for s in seeds]
    tareas1 = [(dataset, K, depth, s, str(directorio), overrides, epsilon, "_eps") for s in seeds]
    resultados = _ejecutar(tareas0 + tareas1, jobs)
    r0 = [json.loads(Path(r).read_text()) for r, _ in resultados[: len(seeds)]]
    r1 = [json.loads(Path(r).read_text()) for r, _ in resultados[len(seeds):]]

    def resumen_de(rs: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "cv_mae_mean": float(np.mean([r["cv_mae_best"] for r in rs])),
            "complexity_mean": float(np.mean([r["complexity"] for r in rs])),
            "n_columns_mean": float(np.mean([r["n_columns"] for r in rs])),
            "seconds_mean": float(np.mean([r["seconds_fit"] for r in rs])),
            "distinct_evaluations_mean": float(np.mean([r["distinct_evaluations"] for r in rs])),
        }

    base = baseline_cv(dataset, seeds)
    res0, res1 = resumen_de(r0), resumen_de(r1)
    dif = res1["cv_mae_mean"] - res0["cv_mae_mean"]
    salida = {
        "dataset": dataset, "K": K, "max_depth": depth, "epsilon": epsilon, "seeds": list(seeds),
        "epsilon_zero": res0, "epsilon_final": res1,
        "per_seed_cv_mae": {"epsilon_zero": [r["cv_mae_best"] for r in r0], "epsilon_final": [r["cv_mae_best"] for r in r1]},
        "difference_cv_mae": dif,
        "baseline_cv": base,
        "stop": bool(dif > epsilon),  # empeora más de ε
    }
    (directorio / dataset).mkdir(parents=True, exist_ok=True)
    (directorio / dataset / "eps_check.json").write_text(json.dumps(salida, indent=1, allow_nan=False))
    return salida


def tabla_epsilon(c: dict[str, Any]) -> str:
    b = c["baseline_cv"]["mean"]
    z, f = c["epsilon_zero"], c["epsilon_final"]
    filas = [
        f"Comprobación 8b — {c['dataset']} (C, K = {c['K']}, profundidad = {c['max_depth']}, semillas {c['seeds']})",
        "",
        "| ε | MAE CV | vs baseline | complejidad | columnas | tiempo (s) | eval. distintas |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for nombre, r in (("0", z), (f"{c['epsilon']:g}", f)):
        filas.append(
            f"| {nombre} | {r['cv_mae_mean']:.4f} | {100 * (b - r['cv_mae_mean']) / b:+.1f} % | "
            f"{r['complexity_mean']:.1f} | {r['n_columns_mean']:.1f} | {r['seconds_mean']:.0f} | "
            f"{r['distinct_evaluations_mean']:.0f} |"
        )
    filas += [
        "",
        f"Baseline (OLS, todos los originales) en CV: {b:.4f}",
        f"MAE CV por semilla con ε = 0: {', '.join(f'{v:.4f}' for v in c['per_seed_cv_mae']['epsilon_zero'])}",
        f"MAE CV por semilla con ε final: {', '.join(f'{v:.4f}' for v in c['per_seed_cv_mae']['epsilon_final'])}",
        f"Diferencia (ε final − ε = 0): {c['difference_cv_mae']:+.4f} (umbral ε = {c['epsilon']:g})",
        "CRITERIO: " + ("EMPEORA más de ε -> PARAR y avisar" if c["stop"] else "no empeora más de ε -> se puede seguir"),
    ]
    return "\n".join(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=sorted(data.DATASETS))
    ap.add_argument("--jobs", type=int, default=max(1, nucleos_fisicos() - 1),
                    help="procesos en paralelo (por defecto, núcleos físicos − 1)")
    ap.add_argument("--out-dir", type=Path, default=DIR_PILOTO)
    ap.add_argument("--common", action="store_true",
                    help="lee los resúmenes de todos los datasets y propone la combinación común")
    ap.add_argument("--eps-check", action="store_true", help="hito 8b: comprobación de ε > 0 (K = 3, profundidad 3)")
    ap.add_argument("--epsilon", type=float, help="ε final del dataset para --eps-check")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.common:
        res = {d: json.loads((args.out_dir / d / "summary.json").read_text()) for d in sorted(data.DATASETS)}
        print(json.dumps(proponer_comun(res), indent=1))
        return
    if args.dataset is None:
        ap.error("--dataset es obligatorio")
    if args.eps_check:
        if args.epsilon is None:
            ap.error("--eps-check necesita --epsilon")
        print(tabla_epsilon(comprobar_epsilon(args.dataset, args.epsilon, args.out_dir, args.jobs)))
    else:
        print(tabla(run_pilot(args.dataset, args.out_dir, args.jobs), args.dataset))
    print(f"\nTiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
