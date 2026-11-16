"""Crea una sola vez las particiones train/validación y muestra el resumen de los datos.

Implementa D1 (RF-01, RF-02). El resumen sirve para contrastar la tabla del §3.
No lee ningún archivo de validación: solo los CSV originales.
"""

from __future__ import annotations

from evofeat.data import DATASETS, make_splits, resumen


def main() -> None:
    for dataset in DATASETS:
        r = resumen(dataset)
        print(f"== {dataset} ==")
        print(f"  filas: {r['filas']}  atributos: {r['atributos']}  objetivo: {r['objetivo']}")
        print(f"  columnas: {', '.join(r['nombres'])}")
        print(f"  faltantes en X: {r['faltantes_X']}  en y: {r['faltantes_y']}  filas duplicadas: {r['filas_duplicadas']}")
        print(f"  y: min={r['y_min']:.4g} max={r['y_max']:.4g} media={r['y_media']:.4g}")
        creada = make_splits(dataset)
        print("  partición: " + ("creada" if creada else "ya existía, no se regenera"))


if __name__ == "__main__":
    main()
