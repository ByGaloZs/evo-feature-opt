"""Una ejecución = dataset × configuración × representación × semilla (RF-09, RF-10, RF-11).

Implementa D1, D10, D11, D22 y §15.4. Es el ÚNICO módulo que llama a
``data.load_validation`` (RNF-02): el conjunto de validación se lee aquí, una sola vez,
después de que el transformador esté ajustado, solo para medir e y e′.

Una ejecución:
1. Carga train y ajusta ``EvoFeatureTransformer`` (nada de validación).
2. Lee validación y mide e (M con los originales) y e′ (M con Xt′), con M = OLS y con
   Ridge, KNN y Random Forest como análisis secundario (D11). En todos los casos M es un
   pipeline escalado + modelo ajustado solo con train (D2, D8).
3. Escribe un JSON con todos los campos de §15.4. Los JSON los escribe solo el código.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from . import data
from .models import SECONDARY_MODELS, make_pipeline_model
from .transformer import EvoFeatureTransformer, RunConfig, load_config

RAIZ = Path(__file__).resolve().parents[2]
DIR_RUNS = RAIZ / "results" / "runs"


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def _evaluar_modelo(nombre: str, seed: int, Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> float:
    """MAE en validación de un modelo (escalado + M) ajustado solo con train."""
    modelo = make_pipeline_model(nombre, seed).fit(Xt, yt)
    return _mae(yv, modelo.predict(Xv))


def _num(x: float) -> float | None:
    """JSON no admite NaN: los valores no definidos (p. ej. en baseline) se guardan como null."""
    return None if x is None or not np.isfinite(x) else float(x)


def config_hash(ruta: str | Path) -> str:
    """Hash SHA-256 del archivo TOML que define el experimento (RNF-01)."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _meta(cfg: RunConfig, dataset: str, seed: int, hash_toml: str) -> dict[str, Any]:
    return {
        "experiment": cfg.name,
        "dataset": dataset,
        "configuration": cfg.configuration,
        "epsilon": cfg.epsilon,
        "representation": cfg.representation,
        "seed": seed,
        "config_hash": hash_toml,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_one(cfg: RunConfig, dataset: str, seed: int, hash_toml: str = "") -> dict[str, Any]:
    """Ejecuta una semilla de un dataset y devuelve el registro (§15.4)."""
    t0 = time.perf_counter()
    cfg = cfg.for_dataset(dataset)  # ε propio del dataset (D17)
    tr = data.load_train(dataset)
    tf = EvoFeatureTransformer(cfg, seed).fit(tr.X, tr.y, feature_names=tr.nombres)
    t_fit = time.perf_counter() - t0

    # Único punto del proyecto que lee la validación (RNF-02)
    va = data.load_validation(dataset)
    Xt_nuevo, Xv_nuevo = tf.transform(tr.X), tf.transform(va.X)
    if not np.isfinite(Xv_nuevo).all():
        raise FloatingPointError("transform produjo valores no finitos en validación")

    e = _evaluar_modelo(cfg.model, seed, tr.X, tr.y, va.X, va.y)
    e_nuevo = _evaluar_modelo(cfg.model, seed, Xt_nuevo, tr.y, Xv_nuevo, va.y)
    secundarios = {
        nombre: {
            "baseline_mae": _evaluar_modelo(nombre, seed, tr.X, tr.y, va.X, va.y),
            "mae": _evaluar_modelo(nombre, seed, Xt_nuevo, tr.y, Xv_nuevo, va.y),
        }
        for nombre in SECONDARY_MODELS
    }
    n_orig = int(len(tf.kept_originals_))
    return {
        "meta": _meta(cfg, dataset, seed, hash_toml),
        "baseline_mae": e,
        "mae": e_nuevo,
        "relative_improvement": (e - e_nuevo) / e,
        "cv_mae_best": _num(tf.cv_mae_best_),
        "evolution_best_fitness": _num(tf.evolution_best_fitness_),
        "n_columns": tf.n_columns,
        "n_originals_kept": n_orig,
        "n_synthesized": len(tf.formulas_),
        "kept_originals": tf.kept_originals_,
        "formulas": tf.formulas_,
        "constant_formulas_dropped": tf.constant_formulas_dropped_,
        "secondary": secundarios,
        "history": tf.history_,
        "hall_of_fame": tf.hall_of_fame_,
        "individuals_generated": tf.individuals_generated_,
        "distinct_evaluations": tf.distinct_evaluations_,
        "evaluations": tf.distinct_evaluations_,
        "cache_hits": tf.cache_hits_,
        "seconds_fit": t_fit,
        "seconds": time.perf_counter() - t0,
    }


def ruta_resultado(cfg: RunConfig, dataset: str, seed: int, dir_runs: Path = DIR_RUNS) -> Path:
    """results/runs/<experimento>/<dataset>/seed_<s>.json (§15.4)."""
    return dir_runs / cfg.name / dataset / f"seed_{seed}.json"


def run_and_save(cfg: RunConfig, dataset: str, seed: int, hash_toml: str = "", dir_runs: Path = DIR_RUNS) -> tuple[Path, bool]:
    """Ejecuta y guarda el JSON; si ya existe, no repite la ejecución. Devuelve (ruta, ejecutada)."""
    ruta = ruta_resultado(cfg, dataset, seed, dir_runs)
    if ruta.exists():
        return ruta, False
    registro = run_one(cfg, dataset, seed, hash_toml)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registro, indent=1, allow_nan=False))
    tmp.replace(ruta)  # escritura atómica: un corte no deja un JSON a medias
    return ruta, True


__all__ = ["run_one", "run_and_save", "ruta_resultado", "config_hash", "load_config"]
