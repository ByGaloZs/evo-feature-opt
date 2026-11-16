"""Lanza las ejecuciones finales de uno o varios experimentos (hito 9).

Implementa RF-10 (D19, D25, §15.4, §15.5). Cada experimento es un TOML de ``configs/``; una
ejecución es dataset × semilla. Los JSON van a ``results/runs/<experimento>/<dataset>/seed_<s>.json``.

- Relanzar el script NO repite las ejecuciones cuyo JSON ya existe.
- Si un JSON existente se generó con otro TOML (hash distinto), el script se detiene: no se
  mezclan resultados de configuraciones distintas bajo el mismo nombre.
- Una ejecución que falla no deja JSON (la escritura es atómica); el resto continúa y el script
  termina con código 1. Al relanzar se reintentan solo las que faltan.
- Una ejecución por proceso (por defecto, núcleos físicos − 1), cada una con un solo hilo de
  álgebra lineal (§15.5). El número de procesos solo cambia el tiempo, no los resultados.

Uso: python scripts/run_experiments.py configs/final_C_tree.toml [configs/final_B.toml ...]
"""

from __future__ import annotations

import os

# un solo hilo de álgebra lineal por proceso (§15.5); debe fijarse antes de importar NumPy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evofeat.experiment import DIR_RUNS, config_hash, ruta_resultado, run_and_save
from evofeat.transformer import RunConfig, load_config

# peso relativo aproximado de una ejecución, solo para lanzar primero las más largas
_PESO_CONFIG = {"baseline": 1.0, "A": 1.2, "B": 7.0, "C": 6.0}
_PESO_DATASET = {"california": 5.0}


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


def peso(cfg: RunConfig, dataset: str) -> float:
    base = _PESO_CONFIG[cfg.configuration] * _PESO_DATASET.get(dataset, 1.0)
    return base * (1.3 if cfg.representation == "gep" else 1.0)


def planificar(rutas_toml: list[Path], dir_runs: Path = DIR_RUNS) -> list[tuple[RunConfig, str, int, str]]:
    """Lista de ejecuciones pendientes (cfg, dataset, semilla, hash), las más largas primero.

    Falla si hay nombres de experimento repetidos o si un JSON existente viene de otro TOML.
    """
    cfgs = [(load_config(str(r)), config_hash(r)) for r in rutas_toml]
    nombres = [c.name for c, _ in cfgs]
    if len(set(nombres)) != len(nombres):
        raise ValueError(f"nombres de experimento repetidos: {nombres}")
    tareas = []
    for cfg, h in cfgs:
        for dataset in cfg.datasets:
            for seed in range(cfg.seed_start, cfg.seed_start + cfg.seed_count):
                ruta = ruta_resultado(cfg, dataset, seed, dir_runs)
                if ruta.exists():
                    previo = json.loads(ruta.read_text())["meta"].get("config_hash")
                    if previo != h:
                        raise RuntimeError(
                            f"{ruta} se generó con otro TOML (hash {str(previo)[:12]} ≠ {h[:12]}): "
                            "no se mezclan resultados; borra los resultados antiguos a mano si procede"
                        )
                    continue
                tareas.append((cfg, dataset, seed, h))
    tareas.sort(key=lambda t: -peso(t[0], t[1]))
    return tareas


def _ejecutar(tarea: tuple[RunConfig, str, int, str], dir_runs: str) -> tuple[str, int, str, float]:
    cfg, dataset, seed, h = tarea
    t0 = time.perf_counter()
    run_and_save(cfg, dataset, seed, h, Path(dir_runs))
    return cfg.name, seed, dataset, time.perf_counter() - t0


def lanzar(rutas_toml: list[Path], jobs: int, dir_runs: Path = DIR_RUNS, verbose: bool = True) -> dict[str, Any]:
    """Ejecuta lo que falte. Devuelve un resumen {pendientes, hechas, fallos}."""
    tareas = planificar(rutas_toml, dir_runs)
    resumen: dict[str, Any] = {"pendientes": len(tareas), "hechas": 0, "fallos": []}
    if not tareas:
        return resumen
    t0 = time.perf_counter()

    def registrar(res: tuple[str, int, str, float]) -> None:
        resumen["hechas"] += 1
        if verbose:
            nombre, seed, dataset, seg = res
            print(f"[{resumen['hechas']}/{len(tareas)}] {nombre} {dataset} semilla {seed}: {seg:.0f} s "
                  f"(transcurrido {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    if jobs <= 1:
        for t in tareas:
            try:
                registrar(_ejecutar(t, str(dir_runs)))
            except Exception:  # noqa: BLE001 - se informa y se continúa
                resumen["fallos"].append((t[0].name, t[1], t[2], traceback.format_exc()))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futuros = {pool.submit(_ejecutar, t, str(dir_runs)): t for t in tareas}
            for f in as_completed(futuros):
                t = futuros[f]
                try:
                    registrar(f.result())
                except Exception:  # noqa: BLE001
                    resumen["fallos"].append((t[0].name, t[1], t[2], traceback.format_exc()))
    return resumen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("configs", nargs="+", type=Path, help="archivos .toml de configs/")
    ap.add_argument("--jobs", type=int, default=max(1, nucleos_fisicos() - 1),
                    help="procesos en paralelo (por defecto, núcleos físicos − 1)")
    ap.add_argument("--out-dir", type=Path, default=DIR_RUNS)
    args = ap.parse_args()
    try:
        resumen = lanzar(args.configs, args.jobs, args.out_dir)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Pendientes al empezar: {resumen['pendientes']}; hechas ahora: {resumen['hechas']}; "
          f"fallos: {len(resumen['fallos'])}")
    for nombre, dataset, seed, tb in resumen["fallos"]:
        print(f"FALLO {nombre} {dataset} semilla {seed}:\n{tb}", file=sys.stderr)
    return 1 if resumen["fallos"] else 0


if __name__ == "__main__":
    sys.exit(main())
