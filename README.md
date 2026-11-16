# evo-feature-opt

Optimización evolutiva de atributos para regresión (práctica de IA bioinspirada, UC3M).
El diseño completo y la fuente de verdad es [`docs/DISENO.md`](docs/DISENO.md); las hipótesis,
escritas antes de lanzar los experimentos finales, están en [`docs/HIPOTESIS.md`](docs/HIPOTESIS.md).

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # todo (≈ 1 min)
pytest -m "not slow"        # sin los tests lentos del bucle evolutivo (≈ 45 s)
```

Los datos originales están en `data/raw/` y no se modifican. Las versiones de las librerías están
fijadas en `pyproject.toml`.

## Cómo reproducir cada resultado

```bash
python scripts/make_splits.py                       # hito 1: particiones train/validación (semilla 2026), una sola vez
python scripts/run_pilot.py --dataset california    # hito 8: rejilla K × profundidad (solo train)
python scripts/run_pilot.py --dataset diabetes
python scripts/run_pilot.py --common                # combinación común a ambos datasets
python scripts/run_pilot.py --dataset diabetes --eps-check --epsilon 1.26      # comprobación de ε > 0 (8b)
python scripts/run_pilot.py --dataset california --eps-check --epsilon 0.0030
python scripts/run_experiments.py configs/final_baseline.toml configs/final_A.toml configs/final_B.toml \
    configs/final_C_tree.toml configs/final_C_gep.toml                         # hito 9: 300 ejecuciones (≈ 1 h)
python scripts/analyze.py results/runs              # hito 10: tablas, tests estadísticos y figuras
```

- Cada experimento es un TOML de `configs/`; cada ejecución escribe un JSON en
  `results/runs/<experimento>/<dataset>/seed_<s>.json` y relanzar el script no repite las ya hechas.
- Si un TOML cambia, `run_experiments.py` se detiene en lugar de mezclar resultados antiguos con nuevos.
- El análisis se escribe en `results/analysis/` (`informe.md` reúne tablas, tests y figuras).
- Los JSON de `results/` los escribe solo el código; no se editan a mano.
- El conjunto de validación solo lo lee `src/evofeat/experiment.py` (RNF-02, hay un test que lo comprueba).
