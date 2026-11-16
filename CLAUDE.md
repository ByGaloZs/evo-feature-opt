# CLAUDE.md

Proyecto: optimización evolutiva de conjuntos de datos para regresión (práctica de IA bioinspirada, UC3M).
El diseño completo está en `docs/DISENO.md` y es la fuente de verdad. Cada decisión tiene un identificador
(D1–D25, RF-xx, RNF-xx) y su justificación. Léelo antes de escribir código.

## Reglas que no se rompen

1. **Validación intocable.** El conjunto de validación solo se lee con `data.load_validation()`, y esa función
   solo se llama desde `experiment.py` para la evaluación final. Nada (escalado, fitness, piloto, elección de
   parámetros) puede usar validación. Hay un test que lo comprueba (RNF-02); no lo desactives.
2. **No decidas por tu cuenta.** Si algo no está especificado en `docs/DISENO.md`, o crees que una decisión
   debería cambiar, para y pregunta. No inventes valores, no cambies parámetros, no añadas funcionalidad que no
   esté en el diseño.
3. **Un hito cada vez** (§15.6 del diseño). Un hito termina cuando pasan sus tests y se hace commit. Después,
   para y resume: qué se hizo, qué tests lo cubren, y cualquier duda o desviación. No empieces el siguiente
   hito sin confirmación.
4. **Reproducibilidad.** Toda la aleatoriedad pasa por un `np.random.Generator` creado a partir de la semilla
   de la ejecución y pasado explícitamente. Prohibido `np.random.*` global y el módulo `random`. Misma semilla,
   mismo resultado.
5. **Dependencias.** Solo las del §15.5 (numpy, scipy, scikit-learn, matplotlib, pytest). Nada más sin preguntar.
   Árboles, GEP y bucle evolutivo son implementación propia con NumPy (§8); no uses DEAP, gplearn ni similares.
6. **Resultados.** Los JSON de `results/` los escribe solo el código; nunca se editan a mano.
   `docs/HIPOTESIS.md` se escribe y se commitea antes de lanzar el hito 9.

## Git

- Los commits van exclusivamente con la identidad de git ya configurada por el usuario. No modifiques
  `user.name` ni `user.email`.
- Sin ningún rastro de Claude: sin `Co-Authored-By`, sin trailers, sin enlaces de sesión y sin mencionar a
  Claude en los mensajes de commit ni en el código.
- No hagas push salvo que se pida.
- Mensajes de commit en español, en imperativo, indicando el hito (p. ej. `Hito 2: funciones protegidas y evaluación de fórmulas`).

## Estilo de código

- Python ≥ 3.11, con anotaciones de tipo.
- Docstrings y comentarios en español.
- Cada módulo empieza con un docstring que indica qué decisiones y requisitos implementa (p. ej. `Implementa D6, D8, D16, D18 (RF-05, RF-07)`),
  para poder rastrear cada línea hasta el diseño.
- Evaluación de fórmulas vectorizada con NumPy; nada de bucles por fila.

## Comandos

```bash
pip install -e ".[dev]"
pytest
python scripts/make_splits.py
python scripts/run_pilot.py --dataset california
python scripts/run_experiments.py configs/<experimento>.toml
python scripts/analyze.py results/runs
```
