Exploratorio, posterior a la validación y **solo con train** (MAE en CV de re-selección, folds nuevos). No cambia los resultados finales: la configuración evaluada en validación sigue siendo `p_mutation = 0,2`.

**Comprobación del control:** las semillas 0, 1, 2 reejecutadas con `p_mutation = 0,2` dan exactamente el mismo MAE en CV que los JSON finales: sí.

| Brazo | n | % CV < 0,5 | mediana CV | Q1 | Q3 | RIC | % transforma AveOccup | % log(1 + \|AveRooms\|) | complejidad | eval. distintas | tiempo (s) | gen. mediana bajo 0,5 (ejecuciones) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pm0.2 (control) | 30 | 70 % (21/30) | 0.4855 | 0.4822 | 0.5106 | 0.0285 | 80 % | 50 % | 17 | 4634 | 114 | 39 (21/30) |
| pm0.5 | 30 | 87 % (26/30) | 0.4828 | 0.4813 | 0.4889 | 0.0076 | 97 % | 67 % | 18.5 | 6256 | 164 | 45 (26/30) |
| pm1 | 30 | 87 % (26/30) | 0.4852 | 0.4820 | 0.4910 | 0.0090 | 97 % | 53 % | 18 | 7120 | 179 | 35 (26/30) |
| GEP (referencia) | 30 | 93 % (28/30) | 0.4834 | 0.4817 | 0.4890 | 0.0074 | 100 % | 47 % | 17 | 6567 | 164 | 29 (28/30) |

Tests **exploratorios, sin corregir por comparaciones múltiples** (bilaterales):

| Brazo | Frente a | Fisher exacto (proporción CV < 0,5), p | Mann-Whitney (MAE en CV), p |
| --- | --- | --- | --- |
| pm0.5 | control (0,2) | 0.209 | 0.186 |
| pm0.5 | GEP | 0.671 | 0.69 |
| pm1 | control (0,2) | 0.209 | 0.455 |
| pm1 | GEP | 0.671 | 0.464 |
