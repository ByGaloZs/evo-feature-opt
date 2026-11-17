"""D23: robustez a la partición, solo en Diabetes (posterior a la validación).

Se ejecutó después de ver los resultados de validación (motivado por la diferencia entre la CV sobre
train y la validación de la partición 2026). Está previsto como opcional desde el diseño.

Diseño: 5 particiones 80/20 nuevas (semillas 2027–2031, distintas de 2026), guardadas en
``results/d23/splits/p<semilla>/`` sin tocar ``data/splits/``. En cada una se ejecutan la línea base y C
con árboles con las semillas 0 a 9, con los MISMOS parámetros que el experimento final (se leen de
``configs/final_baseline.toml`` y ``configs/final_C_tree.toml``: K = 3, profundidad 3, ε = 1,26). Todo se
ajusta solo con el train de esa partición y se evalúa en su validación, reutilizando ``experiment.py``.

Limitación: K, profundidad y ε se eligieron con el train de la partición 2026, que se solapa con la
validación de las nuevas. En Diabetes el piloto no distinguía entre combinaciones, así que el efecto es
despreciable.

Todo se escribe bajo ``results/d23/``: ``splits/``, ``runs/`` (un JSON por ejecución, reanudable) y
``resumen.{md,json}``. No se escribe en ningún otro sitio.

Uso: python scripts/run_d23.py [--dry-run] [--jobs N]
"""

from __future__ import annotations

import os

# un solo hilo de álgebra lineal por proceso (§15.5); debe fijarse antes de importar NumPy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evofeat import data
from evofeat.experiment import run_and_save
from evofeat.transformer import RunConfig, load_config

RAIZ = Path(__file__).resolve().parents[1]
DIR_D23 = RAIZ / "results" / "d23"
DATASET = "diabetes"
SEMILLAS_PARTICION = (2027, 2028, 2029, 2030, 2031)  # distintas de la 2026 original (§15.5)
SEMILLAS_EJECUCION = tuple(range(10))
CONFIGS = {"baseline": RAIZ / "configs" / "final_baseline.toml", "C_tree": RAIZ / "configs" / "final_C_tree.toml"}
DIR_RUNS_REFERENCIA = RAIZ / "results" / "runs"  # partición 2026 (semillas 0–9 de los experimentos finales)
TOL_MEJORA = 1e-9
OPERADORES_CLAVE = {"log(1 + |s6|)": "log(1 + |s6|)", "sex·age": ("(sex * age)", "(age * sex)")}


# ---------------------------------------------------------------------------------------------
# Particiones y plan
# ---------------------------------------------------------------------------------------------

def dir_particion(base: Path, semilla_particion: int) -> Path:
    return Path(base) / "splits" / f"p{semilla_particion}"


def crear_particiones(base: Path = DIR_D23, semillas: tuple[int, ...] = SEMILLAS_PARTICION) -> list[bool]:
    """Crea (si faltan) las particiones 80/20 en ``<base>/splits/p<semilla>/``. No toca data/splits/."""
    return [data.make_splits(DATASET, dir_particion(base, s), semilla=s) for s in semillas]


def nombre_experimento(clave: str, semilla_particion: int) -> str:
    return f"d23_p{semilla_particion}_{clave}"


def config_de(clave: str, semilla_particion: int, overrides: dict[str, Any] | None = None) -> tuple[RunConfig, str]:
    """Configuración final (mismos parámetros) con otro nombre y solo Diabetes; devuelve (cfg, hash)."""
    ruta = CONFIGS[clave]
    cfg = load_config(str(ruta))
    cfg = replace(cfg, name=nombre_experimento(clave, semilla_particion), datasets=(DATASET,),
                  seed_start=0, seed_count=len(SEMILLAS_EJECUCION), **(overrides or {}))
    h = hashlib.sha256(ruta.read_bytes() + f"|d23|{semilla_particion}".encode()).hexdigest()
    return cfg, h


def plan(base: Path = DIR_D23, particiones: tuple[int, ...] = SEMILLAS_PARTICION,
         seeds: tuple[int, ...] = SEMILLAS_EJECUCION, overrides: dict[str, Any] | None = None
         ) -> list[tuple[str, int, int, RunConfig, str]]:
    """Ejecuciones pendientes: (clave, partición, semilla, cfg, hash). Falla si un JSON viene de otro TOML."""
    tareas = []
    for ps in particiones:
        for clave in CONFIGS:
            cfg, h = config_de(clave, ps, overrides)
            for seed in seeds:
                ruta = Path(base) / "runs" / cfg.name / DATASET / f"seed_{seed}.json"
                if ruta.exists():
                    previo = json.loads(ruta.read_text())["meta"].get("config_hash")
                    if previo != h:
                        raise RuntimeError(f"{ruta} se generó con otra configuración (hash distinto): no se mezclan resultados")
                    continue
                tareas.append((clave, ps, seed, cfg, h))
    tareas.sort(key=lambda t: t[0] != "C_tree")  # primero las largas (C), luego la línea base
    return tareas


def estimar(tareas: list[tuple], jobs: int, dir_runs_ref: Path = DIR_RUNS_REFERENCIA) -> dict[str, float]:
    """Tiempo estimado con las ejecuciones finales de Diabetes (mediana de ``seconds`` por configuración)."""
    def mediana(exp: str) -> float:
        s = [json.loads(f.read_text())["seconds"] for f in (dir_runs_ref / exp / DATASET).glob("seed_*.json")]
        return float(np.median(s)) if s else float("nan")

    seg = {"C_tree": mediana("final_C_tree"), "baseline": mediana("final_baseline")}
    cpu = sum(seg[t[0]] for t in tareas)
    return {"runs": len(tareas), "cpu_seconds": cpu, "wall_seconds": cpu / max(1, jobs), "jobs": jobs,
            "seconds_per_C_run": seg["C_tree"], "seconds_per_baseline_run": seg["baseline"]}


def _ejecutar(tarea: tuple[str, int, int, RunConfig, str], base: str) -> tuple[str, int, int, float]:
    clave, ps, seed, cfg, h = tarea
    previo = data.DIR_SPLITS
    data.DIR_SPLITS = dir_particion(Path(base), ps)  # las funciones de carga leen de esta partición
    t0 = time.perf_counter()
    try:
        run_and_save(cfg, DATASET, seed, h, Path(base) / "runs")
    finally:
        data.DIR_SPLITS = previo
    return clave, ps, seed, time.perf_counter() - t0


def lanzar(base: Path = DIR_D23, jobs: int = 1, particiones: tuple[int, ...] = SEMILLAS_PARTICION,
           seeds: tuple[int, ...] = SEMILLAS_EJECUCION, overrides: dict[str, Any] | None = None,
           verbose: bool = True) -> dict[str, Any]:
    """Crea las particiones si faltan y ejecuta lo pendiente. Reanudable."""
    crear_particiones(base, particiones)
    tareas = plan(base, particiones, seeds, overrides)
    resumen: dict[str, Any] = {"pendientes": len(tareas), "hechas": 0, "fallos": []}
    t0 = time.perf_counter()

    def registrar(res: tuple[str, int, int, float]) -> None:
        resumen["hechas"] += 1
        if verbose:
            clave, ps, seed, seg = res
            print(f"[{resumen['hechas']}/{len(tareas)}] partición {ps} {clave} semilla {seed}: {seg:.0f} s "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    if jobs <= 1:
        for t in tareas:
            try:
                registrar(_ejecutar(t, str(base)))
            except Exception:  # noqa: BLE001
                resumen["fallos"].append((t[0], t[1], t[2], traceback.format_exc()))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futuros = {pool.submit(_ejecutar, t, str(base)): t for t in tareas}
            for f in as_completed(futuros):
                t = futuros[f]
                try:
                    registrar(f.result())
                except Exception:  # noqa: BLE001
                    resumen["fallos"].append((t[0], t[1], t[2], traceback.format_exc()))
    return resumen


# ---------------------------------------------------------------------------------------------
# Resumen (descriptivo, sin tests)
# ---------------------------------------------------------------------------------------------

def _cargar(dir_runs: Path, experimento: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rs = [json.loads(f.read_text()) for f in sorted((Path(dir_runs) / experimento / DATASET).glob("seed_*.json"))]
    return sorted([r for r in rs if r["meta"]["seed"] in seeds], key=lambda r: r["meta"]["seed"])


def fila_particion(base_runs: list[dict[str, Any]], c_runs: list[dict[str, Any]]) -> dict[str, float]:
    e = np.array([r["baseline_mae"] for r in c_runs])
    ep = np.array([r["mae"] for r in c_runs])
    cv_base = np.array([r["cv_mae_best"] for r in base_runs])
    return {
        "n": len(c_runs), "e": float(np.median(e)), "cv_baseline": float(np.median(cv_base)),
        "val_over_cv_baseline": float(np.median(e) / np.median(cv_base)),
        "e_prime_median": float(np.median(ep)), "success_pct": float(100 * np.mean(e - ep > TOL_MEJORA * e)),
        "rel_improvement_median": float(np.median([r["relative_improvement"] for r in c_runs])),
        "improvement_points_median": float(np.median(e - ep)),
    }


def estabilidad(c_runs: list[dict[str, Any]], nombres: list[str]) -> dict[str, Any]:
    """Fórmulas más frecuentes, uso de cada original en las fórmulas y reaparición de las fórmulas clave."""
    formulas = [f for r in c_runs for f in r["formulas"]]
    uso = {n: float(np.mean([any(f"{n}" in f.replace("(", " ").replace(")", " ").replace("|", " ").split() for f in r["formulas"])
                             for r in c_runs])) for n in nombres}
    clave: dict[str, Any] = {}
    for etiqueta, patrones in OPERADORES_CLAVE.items():
        pats = (patrones,) if isinstance(patrones, str) else patrones
        clave[etiqueta] = {"runs_with_formula_containing": int(sum(any(p in f for p in pats for f in r["formulas"]) for r in c_runs)),
                           "runs_with_exact_formula": int(sum(any(f in pats for f in r["formulas"]) for r in c_runs))}
    return {"top_formulas": Counter(formulas).most_common(5), "variable_use_fraction": uso, "key_formulas": clave,
            "runs": len(c_runs)}


def resumir(base: Path = DIR_D23, dir_runs_ref: Path = DIR_RUNS_REFERENCIA,
            particiones: tuple[int, ...] = SEMILLAS_PARTICION, seeds: tuple[int, ...] = SEMILLAS_EJECUCION) -> dict[str, Any]:
    nombres = data.load_raw(DATASET).nombres
    entradas: dict[str, tuple[list, list]] = {
        "2026 (referencia)": (_cargar(dir_runs_ref, "final_baseline", seeds), _cargar(dir_runs_ref, "final_C_tree", seeds))}
    for ps in particiones:
        entradas[str(ps)] = (_cargar(Path(base) / "runs", nombre_experimento("baseline", ps), seeds),
                             _cargar(Path(base) / "runs", nombre_experimento("C_tree", ps), seeds))
    filas = {k: fila_particion(b, c) for k, (b, c) in entradas.items() if b and c}
    estab = {k: estabilidad(c, nombres) for k, (b, c) in entradas.items() if c}
    nuevas = [k for k in filas if not k.startswith("2026")]
    mejoras = [filas[k]["rel_improvement_median"] for k in nuevas]
    resumen = {
        "rows": filas, "stability": estab,
        "across_new_partitions": {
            "partitions": len(nuevas),
            "median_e_prime_below_e": int(sum(filas[k]["e_prime_median"] < filas[k]["e"] for k in nuevas)),
            "rel_improvement_median": float(np.median(mejoras)) if mejoras else None,
            "rel_improvement_min": float(min(mejoras)) if mejoras else None,
            "rel_improvement_max": float(max(mejoras)) if mejoras else None,
        },
        "reference_2026": filas.get("2026 (referencia)"),
    }
    todas = [filas[k]["rel_improvement_median"] for k in filas]
    resumen["across_all_including_2026"] = {
        "partitions": len(todas), "median_e_prime_below_e": int(sum(filas[k]["e_prime_median"] < filas[k]["e"] for k in filas)),
        "rel_improvement_median": float(np.median(todas)) if todas else None,
        "rel_improvement_min": float(min(todas)) if todas else None, "rel_improvement_max": float(max(todas)) if todas else None}
    return resumen


def informe(r: dict[str, Any]) -> str:
    p = ["# D23: robustez a la partición en Diabetes (posterior a la validación)", "",
         "Generado por `scripts/run_d23.py`. Descriptivo: con 5 particiones no se hacen tests. Cada partición usa las semillas 0–9; "
         "la 2026 es la de los experimentos finales (semillas 0–9 de `results/runs`).", "",
         "| partición | n | e (validación) | MAE CV baseline (train) | e / CV | mediana e′ | % e′ < e | mejora rel. mediana | mejora (puntos) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, f in r["rows"].items():
        p.append(f"| {k} | {f['n']} | {f['e']:.4g} | {f['cv_baseline']:.4g} | {f['val_over_cv_baseline']:.3f} | {f['e_prime_median']:.4g} | "
                 f"{f['success_pct']:.0f} | {100 * f['rel_improvement_median']:.2f} % | {f['improvement_points_median']:+.3f} |")
    a, t = r["across_new_partitions"], r["across_all_including_2026"]
    if a["rel_improvement_median"] is None or t["rel_improvement_median"] is None:
        return "\n".join(p + ["", "Todavía no hay resultados de las particiones nuevas."]) + "\n"
    p += ["", "## Resumen entre particiones", "",
          f"- Particiones nuevas (5): la mediana de e′ es menor que e en **{a['median_e_prime_below_e']} de {a['partitions']}**; "
          f"mejora relativa mediana entre particiones {100 * a['rel_improvement_median']:.2f} % "
          f"(rango {100 * a['rel_improvement_min']:.2f} % a {100 * a['rel_improvement_max']:.2f} %).",
          f"- Incluida la 2026 ({t['partitions']}): e′ < e en {t['median_e_prime_below_e']} de {t['partitions']}; "
          f"mediana {100 * t['rel_improvement_median']:.2f} % (rango {100 * t['rel_improvement_min']:.2f} % a {100 * t['rel_improvement_max']:.2f} %).", "",
          "## Estabilidad de las fórmulas de C con árboles", ""]
    for k, e in r["stability"].items():
        p += [f"### Partición {k}", "", "Fórmulas más frecuentes:", ""]
        p += [f"- ({c}×) `{f}`" for f, c in e["top_formulas"]]
        p += ["", "Fracción de ejecuciones que usan cada original en sus fórmulas: "
              + ", ".join(f"{n} {v:.2f}" for n, v in e["variable_use_fraction"].items()), "",
              "Fórmulas clave: " + "; ".join(
                  f"{etq}: {x['runs_with_exact_formula']}/{e['runs']} con la fórmula exacta, {x['runs_with_formula_containing']}/{e['runs']} con una fórmula que la contiene"
                  for etq, x in e["key_formulas"].items()), ""]
    return "\n".join(p) + "\n"


def escribir_resumen(base: Path = DIR_D23, dir_runs_ref: Path = DIR_RUNS_REFERENCIA) -> dict[str, Any]:
    r = resumir(base, dir_runs_ref)
    Path(base).mkdir(parents=True, exist_ok=True)
    (Path(base) / "resumen.json").write_text(json.dumps(r, indent=1, allow_nan=False))
    (Path(base) / "resumen.md").write_text(informe(r))
    return r


def _nucleos_fisicos() -> int:
    """Núcleos físicos − 1, reutilizando la función de run_experiments.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_experiments_aux", RAIZ / "scripts" / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return max(1, mod.nucleos_fisicos() - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=None, help="procesos (por defecto, núcleos físicos − 1)")
    ap.add_argument("--dry-run", action="store_true", help="solo muestra el número de ejecuciones y el tiempo estimado")
    ap.add_argument("--summarize", action="store_true", help="solo regenera el resumen con lo que ya hay")
    args = ap.parse_args()
    jobs = args.jobs or _nucleos_fisicos()
    if args.summarize:
        print(informe(escribir_resumen()))
        return 0
    try:
        tareas = plan()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    est = estimar(tareas, jobs)
    print(f"Ejecuciones pendientes: {est['runs']} (5 particiones × [línea base + C con árboles] × 10 semillas = 100 en total)")
    print(f"Tiempo estimado: ≈ {est['cpu_seconds'] / 60:.0f} min de CPU, ≈ {est['wall_seconds'] / 60:.1f} min de reloj con {jobs} procesos "
          f"(C: ≈ {est['seconds_per_C_run']:.0f} s por ejecución, línea base: ≈ {est['seconds_per_baseline_run']:.0f} s)")
    if args.dry_run:
        return 0
    r = lanzar(DIR_D23, jobs)
    print(f"Hechas ahora: {r['hechas']}; fallos: {len(r['fallos'])}")
    for clave, ps, seed, tb in r["fallos"]:
        print(f"FALLO partición {ps} {clave} semilla {seed}:\n{tb}", file=sys.stderr)
    print(informe(escribir_resumen()))
    return 1 if r["fallos"] else 0


if __name__ == "__main__":
    sys.exit(main())
