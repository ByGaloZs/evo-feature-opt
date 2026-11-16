"""Carga y validación de los CSV y partición train/validación.

Implementa D1, D3 (detección), D4, D5 (RF-01, RF-02, RNF-02).

- Los CSV originales (``data/raw``) no se tocan.
- La partición 80/20 se hace una sola vez con semilla fija (2026), se guarda en
  ``data/splits`` y, si ya existe, se lee sin regenerarla.
- ``load_validation`` es la ÚNICA puerta de acceso al conjunto de validación y
  solo debe llamarla ``experiment.py`` en la evaluación final (RNF-02).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"
DIR_SPLITS = RAIZ / "data" / "splits"

SEMILLA_PARTICION = 2026  # §15.5: igual para todos los experimentos
FRACCION_VALIDACION = 0.2  # D1: 80/20

# Dataset -> (archivo CSV, columna objetivo). La columna objetivo se declara
# explícitamente y se comprueba al cargar; no se adivina.
DATASETS: dict[str, tuple[str, str]] = {
    "diabetes": ("Diabetes.csv", "target"),
    "california": ("California.csv", "MedHouseVal"),
}


@dataclass(frozen=True)
class Datos:
    """Matriz de atributos, objetivo y nombres de columnas de un dataset."""

    X: np.ndarray  # (m, n) float64
    y: np.ndarray  # (m,) float64
    nombres: list[str]  # nombres de las n columnas de X
    objetivo: str  # nombre de la columna objetivo


def _leer_csv(ruta: Path) -> tuple[list[str], np.ndarray]:
    """Lee un CSV numérico. Los huecos se leen como NaN; texto no numérico falla."""
    with open(ruta, newline="") as f:
        lector = csv.reader(f)
        cabecera = next(lector)
        filas = []
        for i, fila in enumerate(lector, start=2):
            if len(fila) != len(cabecera):
                raise ValueError(f"{ruta.name}: la línea {i} tiene {len(fila)} campos, se esperaban {len(cabecera)}")
            filas.append([float(c) if c.strip() != "" else np.nan for c in fila])
    return cabecera, np.asarray(filas, dtype=np.float64)


def _separar(cabecera: list[str], M: np.ndarray, objetivo: str, origen: str) -> Datos:
    """Separa X de y; el objetivo nunca aparece en X (D4)."""
    if objetivo not in cabecera:
        raise ValueError(f"{origen}: no existe la columna objetivo '{objetivo}' (columnas: {cabecera})")
    j = cabecera.index(objetivo)
    nombres = [c for k, c in enumerate(cabecera) if k != j]
    X = np.delete(M, j, axis=1)
    return Datos(X=X, y=M[:, j].copy(), nombres=nombres, objetivo=objetivo)


def _config(dataset: str) -> tuple[str, str]:
    if dataset not in DATASETS:
        raise KeyError(f"dataset desconocido '{dataset}'; disponibles: {sorted(DATASETS)}")
    return DATASETS[dataset]


def load_raw(dataset: str) -> Datos:
    """Carga el CSV original completo (sin partir). Solo para splits y resumen."""
    archivo, objetivo = _config(dataset)
    cabecera, M = _leer_csv(DIR_RAW / archivo)
    return _separar(cabecera, M, objetivo, archivo)


def resumen(dataset: str) -> dict:
    """Resumen para contrastar con la tabla del §3 (filas, columnas, objetivo, faltantes)."""
    d = load_raw(dataset)
    return {
        "dataset": dataset,
        "filas": int(d.X.shape[0]),
        "atributos": int(d.X.shape[1]),
        "nombres": d.nombres,
        "objetivo": d.objetivo,
        "faltantes_X": int(np.isnan(d.X).sum()),
        "faltantes_y": int(np.isnan(d.y).sum()),
        "y_min": float(np.nanmin(d.y)),
        "y_max": float(np.nanmax(d.y)),
        "y_media": float(np.nanmean(d.y)),
        "filas_duplicadas": int(d.X.shape[0] - len(np.unique(np.column_stack([d.X, d.y]), axis=0))),
    }


def _ruta_split(dataset: str, parte: str) -> Path:
    return DIR_SPLITS / f"{dataset}_{parte}.csv"


def _guardar(ruta: Path, nombres: list[str], M: np.ndarray) -> None:
    # %.17g reproduce el float64 exactamente al releer
    np.savetxt(ruta, M, delimiter=",", header=",".join(nombres), comments="", fmt="%.17g")


def make_splits(dataset: str, directorio: Path | None = None, semilla: int = SEMILLA_PARTICION) -> bool:
    """Crea la partición 80/20 si no existe (D1, RF-02). Devuelve True si la creó.

    Si los dos archivos ya existen no se regeneran: así todos los experimentos
    miden contra el mismo conjunto de validación.
    """
    destino = DIR_SPLITS if directorio is None else directorio
    r_train, r_val = destino / f"{dataset}_train.csv", destino / f"{dataset}_val.csv"
    if r_train.exists() and r_val.exists():
        return False
    archivo, objetivo = _config(dataset)
    cabecera, M = _leer_csv(DIR_RAW / archivo)
    _separar(cabecera, M, objetivo, archivo)  # valida que la columna objetivo existe
    rng = np.random.default_rng(semilla)
    perm = rng.permutation(M.shape[0])
    n_val = int(round(FRACCION_VALIDACION * M.shape[0]))
    destino.mkdir(parents=True, exist_ok=True)
    _guardar(r_val, cabecera, M[perm[:n_val]])
    _guardar(r_train, cabecera, M[perm[n_val:]])
    return True


def _cargar_split(dataset: str, parte: str) -> Datos:
    archivo, objetivo = _config(dataset)
    ruta = _ruta_split(dataset, parte)
    if not ruta.exists():
        raise FileNotFoundError(f"{ruta} no existe; ejecuta scripts/make_splits.py")
    cabecera, M = _leer_csv(ruta)
    return _separar(cabecera, M, objetivo, ruta.name)


def load_train(dataset: str) -> Datos:
    """Conjunto de entrenamiento (Xt, yt)."""
    return _cargar_split(dataset, "train")


def load_validation(dataset: str) -> Datos:
    """Conjunto de validación (Xv, yv).

    SOLO la llama experiment.py para la evaluación final (RNF-02). Ningún otro
    módulo (escalado, fitness, piloto, elección de parámetros) debe usarla.
    """
    return _cargar_split(dataset, "val")
