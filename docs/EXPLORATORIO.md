# Análisis exploratorio: tasa de mutación de los árboles

Fecha de redacción: **2026-11-17**. Este archivo se escribe y se commitea **antes de lanzar ninguna ejecución** del experimento. No se edita
la parte de planteamiento (motivación, H6 y diseño) después de ver resultados; solo se añade al final la sección «Resultados e interpretación».

Es un análisis **exploratorio**. Se decide **después de ver los resultados de validación** y el diagnóstico de la sección 5.2 de
`docs/RESULTADOS.md`, pero **solo usa train**: no lee validación ni calcula e′. **No cambia el sistema ni los resultados finales**: la configuración
evaluada en validación sigue siendo `p_mutation = 0,2`. No se toca `docs/HIPOTESIS.md`. Se presenta salga lo que salga.

## Motivación (sección 5.2 de `docs/RESULTADOS.md`)

En California, 11 de las 30 ejecuciones finales de C con árboles son malas (e′ > 0,4906). Su e′ malo ya se ve en la CV sobre train
(MAE en CV mediano 0,5165 frente a 0,4838 en las buenas; correlación de Spearman entre e′ y el MAE en CV de 0,97): es un fallo de búsqueda, no de
generalización. Con GEP, 28 de 30 ejecuciones bajan de 0,50 en MAE en CV (generación mediana 29) frente a 21 de 30 con árboles (generación mediana 39),
y el rango intercuartílico de e′ es 0,0074 frente a 0,0277.

Con las tasas de los TOML finales, el número esperado de operaciones de mutación sobre fórmulas por individuo es 0,20 en árboles (`p_mutation = 0,2`:
se muta un árbol en el 20 % de los individuos) frente a 2,58 en GEP (tres genes con mutación puntual 0,044 por símbolo, más IS y RIS), unas 13 veces
más. La hipótesis de causa de la sección 5.2, no probada, es que esta asimetría explica la diferencia de fiabilidad entre las dos representaciones.

## H6 (exploratoria)

Con `p_mutation` mayor, **aumenta la proporción de ejecuciones con MAE en CV < 0,5** y **disminuye el RIC del MAE en CV**, acercándose a GEP (28/30).
Si no ocurre, la asimetría de mutación no explica la diferencia entre árboles y GEP.

## Diseño

- Solo California, configuración C con árboles. Todo igual que en los experimentos finales (K = 3, profundidad 3, ε = 0,0030, presupuesto, selección,
  re-selección), salvo `p_mutation`.
- Brazos: `p_mutation` = 0,5 y 1,0, con las semillas 0 a 29 (60 ejecuciones nuevas).
- Control (`p_mutation` = 0,2): se reutiliza el MAE en CV de re-selección de las 30 ejecuciones finales de C con árboles. No se repite, porque la
  simulación es determinista. Para comprobarlo, se vuelven a ejecutar las semillas 0, 1 y 2 con `p_mutation` = 0,2 en este mismo modo y se verifica que
  el MAE en CV coincide exactamente con el de los JSON finales. Si no coincide, se para y se avisa.
- Referencia: el MAE en CV de re-selección de las 30 ejecuciones finales de C con GEP.
- **Solo train.** La métrica es el MAE en CV de re-selección (folds nuevos), igual que en el piloto. No se lee validación ni se calcula e′. Se reutiliza la
  maquinaria del piloto (ajuste del transformador solo con train), no `run_experiments.py`. Un test comprueba que el script no llama a `load_validation`.
- Resultados en `results/explore_mutation/`, separados del resto. Script reanudable.
- Antes de lanzar, se informa del número de ejecuciones y del tiempo estimado.

**Qué se mide, por brazo (0,2 control, 0,5, 1,0) y GEP como referencia:**
- % de ejecuciones con MAE en CV < 0,5 (el umbral de la sección 5.2).
- Mediana, Q1, Q3 y RIC del MAE en CV.
- % de ejecuciones con alguna fórmula que transforma AveOccup y % con `log(1 + |AveRooms|)`, con la misma heurística de la sección 5.2.
- Complejidad mediana, evaluaciones distintas y tiempo mediano.
- Generación mediana en la que el mejor MAE en CV baja de 0,5.

**Tests exploratorios, sin corregir por comparaciones múltiples y marcados como tales:**
- Fisher exacto bilateral sobre la proporción con CV < 0,5: cada brazo frente al control y frente a GEP.
- Mann-Whitney bilateral sobre el MAE en CV: cada brazo frente al control y frente a GEP.

**Limitación del diseño, anotada de antemano.** En el código de los árboles, `p_mutation` es la probabilidad de mutar **un** árbol por individuo. Incluso con
`p_mutation = 1,0` hay como mucho una operación de mutación de fórmula por individuo (1,0), por debajo de las 2,58 de GEP. Si los árboles mejoran con
1,0 pero no llegan a GEP, no se puede distinguir entre «falta aún tasa de mutación» y «otra diferencia entre las representaciones».

## Resultados e interpretación

Ejecutado el 2026-11-17: 60 ejecuciones, 0 fallos. Antes de lanzar los brazos, las semillas 0, 1 y 2 reejecutadas con `p_mutation` = 0,2 dieron exactamente el
mismo MAE en CV que los JSON finales (el control se reutiliza con garantías). Tabla y tests en `results/explore_mutation/resumen.md` y en la sección 5.4 de
`docs/RESULTADOS.md`.

**Lo observado (solo train).**

| | % CV < 0,5 | RIC del MAE en CV | % transforma AveOccup |
| --- | --- | --- | --- |
| `p_mutation` 0,2 (control) | 70 % (21/30) | 0,0285 | 80 % |
| 0,5 | 87 % (26/30) | 0,0076 | 97 % |
| 1,0 | 87 % (26/30) | 0,0090 | 97 % |
| GEP | 93 % (28/30) | 0,0074 | 100 % |

El patrón descriptivo es el que predecía H6: la proporción con CV < 0,5 sube y se acerca a GEP, y el RIC cae hasta el nivel de GEP. Lo que desaparece es la cola
mala (Q3 de 0,5106 a 0,4889 y 0,4910), no el centro (medianas 0,4855, 0,4828, 0,4852; GEP 0,4834).

**Qué no se puede afirmar.** Los tests no detectan diferencia con el control (Fisher p = 0,209 en ambos brazos; Mann-Whitney p = 0,186 y 0,455) y tampoco con
GEP (Fisher p = 0,671; Mann-Whitney p = 0,690 y 0,464), sin corregir; con 30 ejecuciones por brazo, una diferencia de 5 ejecuciones (21 frente a 26) no es
detectable. No hay efecto dosis-respuesta entre 0,5 y 1,0. Además, más mutación va acompañada de más evaluaciones distintas (4 634 → 6 256 y 7 120; GEP 6 567) y más tiempo
(114 → 164 y 179 s; GEP 164 s), así que el cambio no es solo de tasa de mutación.

**Interpretación.** El resultado es compatible con que la asimetría de mutación explique, al menos en parte, la menor fiabilidad de los árboles, pero no lo demuestra.
Tampoco descarta una diferencia residual entre representaciones: con `p_mutation` = 1,0 sigue habiendo una sola operación de mutación por individuo (frente a 2,58 en GEP) y
los brazos quedan algo por debajo de GEP (87 % frente a 93 %). Otras diferencias que podrían contribuir, sin probarlas: el cruce (en GEP se aplica a cada gen en cada pareja; en
árboles, a un solo árbol y con puerta de 0,8) y la recombinación de genes, y que los cromosomas de GEP tienen símbolos no expresados que permiten cambios neutros.

**Esto no cambia los resultados finales:** la configuración evaluada en validación sigue siendo `p_mutation` = 0,2.
