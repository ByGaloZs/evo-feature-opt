# Resultados

Generado por `scripts/analyze.py` a partir de `results/runs`; no se edita a mano.

## Contraste de hipótesis (`docs/HIPOTESIS.md`)

- **H1** — En California, C mejora claramente (e′ < e). → **apoyada**. mejora relativa mediana 7.92 %, Wilcoxon unilateral p = 1.9e-06.
- **H2** — En Diabetes, la mejora es pequeña o nula. → **interpretación manual**. mejora relativa mediana 1.28 %, Wilcoxon unilateral p = 0.0053; el diseño no fija un umbral para «pequeña»: interpretación manual.
- **H3** — B aporta más que A en ambos datasets. → **apoyada**. California: mediana e′ B = 0.4731, A = 0.5219, p = 1.03e-08; Diabetes: mediana e′ B = 38.67, A = 39.78, p = 0.000863.
- **H4** — Sin diferencia significativa en e′ entre árboles y GEP; si la hay, en la velocidad de convergencia. → **apoyada**. California: e′ árbol = 0.4806, GEP = 0.4786, p = 0.137; gen90 árbol = 59, GEP = 61, p = 0.796; Diabetes: e′ árbol = 38.72, GEP = 38.61, p = 0.912; gen90 árbol = 71, GEP = 68, p = 0.371. (no rechazar la igualdad no prueba que sea cierta)
- **H5** — En California, B obtiene un e′ menor que C (coste de exigir s < n). → **apoyada**. mediana e′ B = 0.4731, C = 0.4806 (coste +0.007495), p = 0.000476; C usa n − 1 = 7 columnas en el 100 % de las ejecuciones.

## e frente a e′ por dataset y configuración

| Dataset | Configuración | n | e | mediana e′ | Q1 | Q3 | RIC | % e′ < e | mejora rel. mediana | columnas | sintetizadas | MAE CV mediana | e′ − MAE CV | tiempo (s) | eval. distintas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| California | Línea base | 30 | 0.5219 | 0.5219 | 0.5219 | 0.5219 | 0 | 0 | -0.00 % | 8 | 0 | 0.5363 | -0.01445 | 31 | 0 |
| California | A (selección) | 30 | 0.5219 | 0.5219 | 0.5219 | 0.5219 | 0 | 0 | -0.01 % | 7 | 0 | 0.5363 | -0.01437 | 34 | 202 |
| California | B (síntesis) | 30 | 0.5219 | 0.4731 | 0.4701 | 0.4802 | 0.01012 | 90 | 9.35 % | 11 | 3 | 0.4785 | -0.004828 | 98 | 1535 |
| California | C (árbol) | 30 | 0.5219 | 0.4806 | 0.4755 | 0.5032 | 0.02766 | 97 | 7.92 % | 7 | 3 | 0.4855 | -0.006344 | 144 | 4634 |
| California | C (GEP) | 30 | 0.5219 | 0.4786 | 0.4744 | 0.4818 | 0.007436 | 97 | 8.30 % | 7 | 2 | 0.4834 | -0.006334 | 193 | 6567 |
| Diabetes | Línea base | 30 | 39.22 | 39.22 | 39.22 | 39.22 | 0 | 0 | 0.00 % | 10 | 0 | 46.22 | -6.997 | 1 | 0 |
| Diabetes | A (selección) | 30 | 39.22 | 39.78 | 39.24 | 40 | 0.759 | 13 | -1.43 % | 6 | 0 | 45.76 | -6.109 | 4 | 538 |
| Diabetes | B (síntesis) | 30 | 39.22 | 38.67 | 38.01 | 39.37 | 1.359 | 67 | 1.41 % | 12 | 2 | 45.3 | -6.375 | 8 | 1198 |
| Diabetes | C (árbol) | 30 | 39.22 | 38.72 | 38.26 | 39.3 | 1.037 | 73 | 1.28 % | 8 | 2 | 44.78 | -5.849 | 23 | 3592 |
| Diabetes | C (GEP) | 30 | 39.22 | 38.61 | 38.23 | 39.43 | 1.201 | 67 | 1.57 % | 8 | 2 | 44.7 | -5.889 | 24 | 3800 |

## Pruebas estadísticas

Nivel de significación α = 0.05. Los p-valores no se corrigen entre hipótesis distintas.

### Wilcoxon de una muestra sobre e′ − e (unilateral: e′ < e)

| Dataset | Configuración | mediana e′ − e | mejora rel. mediana | p | significativo |
| --- | --- | --- | --- | --- | --- |
| California | A (selección) | 4.705e-05 | -0.01 % | 1 | no |
| California | B (síntesis) | -0.0488 | 9.35 % | 0.00021 | sí |
| California | C (árbol) | -0.04131 | 7.92 % | 1.9e-06 | sí |
| California | C (GEP) | -0.04331 | 8.30 % | 9.6e-07 | sí |
| Diabetes | A (selección) | 0.5624 | -1.43 % | 1 | no |
| Diabetes | B (síntesis) | -0.5513 | 1.41 % | 0.0202 | sí |
| Diabetes | C (árbol) | -0.5027 | 1.28 % | 0.0053 | sí |
| Diabetes | C (GEP) | -0.6162 | 1.57 % | 0.046 | sí |

### Kruskal-Wallis entre A, B y C (árbol) sobre e′, y Mann-Whitney por pares con Holm

| Dataset | Contraste | medianas / estadístico | p crudo | p (Holm) |  | significativo |
| --- | --- | --- | --- | --- | --- | --- |
| California | Kruskal-Wallis | H = 49.7 |  | 1.58e-11 |  | sí |
| California | Mann-Whitney A-B | 0.5219 vs 0.4731 | 2.05e-08 | 4.11e-08 |  | sí |
| California | Mann-Whitney A-C | 0.5219 vs 0.4806 | 6.02e-11 | 1.81e-10 |  | sí |
| California | Mann-Whitney B-C | 0.4731 vs 0.4806 | 0.000952 | 0.000952 |  | sí |
| Diabetes | Kruskal-Wallis | H = 18.6 |  | 9.32e-05 |  | sí |
| Diabetes | Mann-Whitney A-B | 39.78 vs 38.67 | 0.00173 | 0.00345 |  | sí |
| Diabetes | Mann-Whitney A-C | 39.78 vs 38.72 | 1.63e-05 | 4.9e-05 |  | sí |
| Diabetes | Mann-Whitney B-C | 38.67 vs 38.72 | 0.912 | 0.912 |  | no |

Los pares solo son interpretables si Kruskal-Wallis es significativo (D22).

### H3: B frente a A (unilateral: e′ de B < e′ de A)

| Dataset | mediana e′ B | mediana e′ A | p | significativo |
| --- | --- | --- | --- | --- |
| California | 0.4731 | 0.5219 | 1.03e-08 | sí |
| Diabetes | 38.67 | 39.78 | 0.000863 | sí |

### H4 (D25): C con árboles frente a C con GEP (bilateral)

| Dataset | medida | mediana árbol | mediana GEP | p | significativo |
| --- | --- | --- | --- | --- | --- |
| California | e′ | 0.4806 | 0.4786 | 0.137 | no |
| California | columnas | 7 | 7 | 0.334 | no |
| California | nodos de fórmula (total) | 10 | 10.5 | 0.894 | no |
| California | generación al 90 % de la mejora | 59 | 61 | 0.796 | no |
| California | tiempo (s) | 144.1 | 192.8 | 6.07e-11 | sí |
| California | evaluaciones distintas | 4634 | 6567 | 7.38e-11 | sí |
| Diabetes | e′ | 38.72 | 38.61 | 0.912 | no |
| Diabetes | columnas | 8 | 8 | 0.676 | no |
| Diabetes | nodos de fórmula (total) | 10.5 | 10.5 | 0.494 | no |
| Diabetes | generación al 90 % de la mejora | 71 | 68.5 | 0.371 | no |
| Diabetes | tiempo (s) | 22.6 | 23.96 | 0.0103 | sí |
| Diabetes | evaluaciones distintas | 3592 | 3800 | 0.126 | no |

### H5: B frente a C en California (unilateral: e′ de B < e′ de C)

| Dataset | mediana e′ B | mediana e′ C | coste de s < n (C − B) | p | C usa n − 1 columnas | significativo |
| --- | --- | --- | --- | --- | --- | --- |
| California | 0.4731 | 0.4806 | +0.007495 | 0.000476 | 100 % | sí |
| Diabetes | 38.67 | 38.72 | +0.04862 | 0.456 | 43 % | no |

## Modelos secundarios (D11)

| Dataset | Configuración | Modelo | mediana e | mediana e′ | Q1 | Q3 | % e′ < e | mejora rel. mediana | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| California | A (selección) | Ridge | 0.5219 | 0.5219 | 0.5219 | 0.5219 | 0 | -0.01 % | 1 |
| California | A (selección) | KNN | 0.426 | 0.3963 | 0.3963 | 0.3963 | 93 | 6.98 % | 3.55e-05 |
| California | A (selección) | Random Forest | 0.3303 | 0.3282 | 0.3277 | 0.3287 | 100 | 0.60 % | 9.31e-10 |
| California | B (síntesis) | Ridge | 0.5219 | 0.473 | 0.4699 | 0.48 | 90 | 9.36 % | 0.00021 |
| California | B (síntesis) | KNN | 0.426 | 0.4357 | 0.4183 | 0.442 | 43 | -2.26 % | 0.971 |
| California | B (síntesis) | Random Forest | 0.3303 | 0.3326 | 0.3312 | 0.3339 | 20 | -0.78 % | 0.987 |
| California | C (árbol) | Ridge | 0.5219 | 0.4806 | 0.4755 | 0.5032 | 97 | 7.92 % | 1.9e-06 |
| California | C (árbol) | KNN | 0.426 | 0.4129 | 0.4025 | 0.4197 | 80 | 3.09 % | 0.000643 |
| California | C (árbol) | Random Forest | 0.3303 | 0.3298 | 0.3273 | 0.3316 | 57 | 0.26 % | 0.452 |
| California | C (GEP) | Ridge | 0.5219 | 0.4786 | 0.4744 | 0.4818 | 97 | 8.30 % | 9.6e-07 |
| California | C (GEP) | KNN | 0.426 | 0.4156 | 0.409 | 0.4208 | 90 | 2.45 % | 7.46e-06 |
| California | C (GEP) | Random Forest | 0.3303 | 0.3291 | 0.328 | 0.3309 | 63 | 0.45 % | 0.103 |
| Diabetes | A (selección) | Ridge | 39.36 | 39.8 | 39.26 | 40 | 30 | -1.11 % | 1 |
| Diabetes | A (selección) | KNN | 39.95 | 38.89 | 36.7 | 39.89 | 83 | 2.66 % | 0.000353 |
| Diabetes | A (selección) | Random Forest | 40.96 | 41.43 | 41.04 | 41.69 | 17 | -0.98 % | 0.988 |
| Diabetes | B (síntesis) | Ridge | 39.36 | 38.79 | 38.35 | 39.55 | 67 | 1.45 % | 0.0202 |
| Diabetes | B (síntesis) | KNN | 39.95 | 40.53 | 39.93 | 41.57 | 27 | -1.44 % | 0.999 |
| Diabetes | B (síntesis) | Random Forest | 40.96 | 41.2 | 40.87 | 41.65 | 20 | -0.71 % | 0.999 |
| Diabetes | C (árbol) | Ridge | 39.36 | 38.75 | 38.39 | 39.31 | 77 | 1.56 % | 0.00109 |
| Diabetes | C (árbol) | KNN | 39.95 | 41.9 | 39.77 | 44.39 | 27 | -4.86 % | 1 |
| Diabetes | C (árbol) | Random Forest | 40.96 | 41.75 | 41.25 | 42.31 | 23 | -1.96 % | 1 |
| Diabetes | C (GEP) | Ridge | 39.36 | 38.62 | 38.33 | 39.4 | 73 | 1.89 % | 0.0139 |
| Diabetes | C (GEP) | KNN | 39.95 | 41.96 | 41.01 | 43.34 | 20 | -5.03 % | 1 |
| Diabetes | C (GEP) | Random Forest | 40.96 | 41.41 | 40.73 | 42.23 | 27 | -1.31 % | 0.993 |

## Fórmulas y atributos

### California

Fracción de ejecuciones que conservan cada atributo original (los que menos aparecen son los más eliminados); B conserva todos por construcción:

| Atributo | A | C árbol | C GEP |
| --- | --- | --- | --- |
| MedInc | 1.00 | 0.90 | 0.97 |
| HouseAge | 1.00 | 0.83 | 0.90 |
| AveRooms | 1.00 | 0.20 | 0.07 |
| AveBedrms | 1.00 | 0.27 | 0.13 |
| Population | 0.07 | 0.07 | 0.27 |
| AveOccup | 0.93 | 0.50 | 0.37 |
| Latitude | 1.00 | 0.87 | 1.00 |
| Longitude | 1.00 | 0.90 | 1.00 |

Fracción de ejecuciones cuyas fórmulas usan cada atributo original:

| Atributo | B | C árbol | C GEP |
| --- | --- | --- | --- |
| MedInc | 0.03 | 0.23 | 0.07 |
| HouseAge | 0.10 | 0.20 | 0.17 |
| AveRooms | 0.90 | 0.80 | 0.80 |
| AveBedrms | 0.27 | 0.20 | 0.10 |
| Population | 0.07 | 0.10 | 0.03 |
| AveOccup | 1.00 | 0.87 | 1.00 |
| Latitude | 0.30 | 0.43 | 0.23 |
| Longitude | 0.23 | 0.30 | 0.10 |

Operadores en las fórmulas finales (recuento total):

| Operador | B | C árbol | C GEP |
| --- | --- | --- | --- |
| add | 20 | 35 | 21 |
| sub | 14 | 24 | 24 |
| mul | 38 | 17 | 11 |
| div | 4 | 8 | 11 |
| sqrt | 52 | 42 | 42 |
| log | 80 | 56 | 92 |

Tamaño de las fórmulas y diversidad:

| Configuración | fórmulas | distintas | nodos (mediana) | nodos (máx.) |
| --- | --- | --- | --- | --- |
| B | 90 | 44 | 4 | 13 |
| C árbol | 74 | 56 | 5 | 12 |
| C GEP | 68 | 44 | 5 | 11 |

Fórmulas más frecuentes — B (síntesis):

- (18×) `log(1 + |AveRooms|)`
- (9×) `log(1 + |AveOccup|)`
- (8×) `sqrt(|(AveOccup * AveOccup)|)`
- (7×) `(sqrt(|AveOccup|) * sqrt(|AveOccup|))`
- (3×) `sqrt(|AveOccup|)`

Fórmulas más frecuentes — C (árbol):

- (8×) `log(1 + |AveRooms|)`
- (5×) `(AveOccup / sqrt(|AveOccup|))`
- (3×) `sqrt(|AveOccup|)`
- (3×) `log(1 + |(AveOccup - log(1 + |AveOccup|))|)`
- (2×) `sqrt(|Latitude|)`

Fórmulas más frecuentes — C (GEP):

- (5×) `log(1 + |AveRooms|)`
- (5×) `log(1 + |log(1 + |AveRooms|)|)`
- (4×) `log(1 + |log(1 + |(AveOccup - log(1 + |AveOccup|))|)|)`
- (3×) `sqrt(|AveOccup|)`
- (3×) `log(1 + |log(1 + |(log(1 + |AveOccup|) - AveOccup)|)|)`

### Diabetes

Fracción de ejecuciones que conservan cada atributo original (los que menos aparecen son los más eliminados); B conserva todos por construcción:

| Atributo | A | C árbol | C GEP |
| --- | --- | --- | --- |
| age | 0.00 | 0.00 | 0.10 |
| sex | 1.00 | 1.00 | 0.97 |
| bmi | 1.00 | 0.97 | 0.97 |
| bp | 1.00 | 1.00 | 0.97 |
| s1 | 0.47 | 0.50 | 0.53 |
| s2 | 0.17 | 0.37 | 0.57 |
| s3 | 0.83 | 0.63 | 0.53 |
| s4 | 0.07 | 0.13 | 0.10 |
| s5 | 1.00 | 0.97 | 0.97 |
| s6 | 0.17 | 0.43 | 0.43 |

Fracción de ejecuciones cuyas fórmulas usan cada atributo original:

| Atributo | B | C árbol | C GEP |
| --- | --- | --- | --- |
| age | 0.57 | 0.70 | 0.60 |
| sex | 0.53 | 0.60 | 0.60 |
| bmi | 0.63 | 0.63 | 0.70 |
| bp | 0.20 | 0.17 | 0.13 |
| s1 | 0.23 | 0.17 | 0.20 |
| s2 | 0.17 | 0.27 | 0.10 |
| s3 | 0.03 | 0.20 | 0.30 |
| s4 | 0.20 | 0.20 | 0.20 |
| s5 | 0.33 | 0.33 | 0.53 |
| s6 | 0.63 | 0.83 | 0.77 |

Operadores en las fórmulas finales (recuento total):

| Operador | B | C árbol | C GEP |
| --- | --- | --- | --- |
| add | 44 | 40 | 45 |
| sub | 13 | 20 | 26 |
| mul | 28 | 22 | 24 |
| div | 9 | 21 | 14 |
| sqrt | 25 | 36 | 41 |
| log | 25 | 29 | 35 |

Tamaño de las fórmulas y diversidad:

| Configuración | fórmulas | distintas | nodos (mediana) | nodos (máx.) |
| --- | --- | --- | --- | --- |
| B | 52 | 47 | 6 | 13 |
| C árbol | 65 | 58 | 5 | 13 |
| C GEP | 58 | 53 | 6 | 12 |

Fórmulas más frecuentes — B (síntesis):

- (4×) `log(1 + |s6|)`
- (2×) `(age * sex)`
- (2×) `(sex / s6)`
- (1×) `sqrt(|((bmi + s6) + s6)|)`
- (1×) `((age + log(1 + |s1|)) * ((age + sex) + sex))`

Fórmulas más frecuentes — C (árbol):

- (4×) `log(1 + |s6|)`
- (3×) `(sex * age)`
- (2×) `sqrt(|(s6 + bmi)|)`
- (2×) `log(1 + |age|)`
- (1×) `(((s5 / s3) * (s6 - age)) * sqrt(|s2|))`

Fórmulas más frecuentes — C (GEP):

- (5×) `log(1 + |s6|)`
- (2×) `sqrt(|bmi|)`
- (1×) `(((sex + s6) * age) - (s5 + (bmi + s1)))`
- (1×) `(s6 - s5)`
- (1×) `log(1 + |sqrt(|sqrt(|(s6 + bmi)|)|)|)`


## Figuras

- `figuras/cajas_e_prime.png`: diagramas de caja de e′ con la línea de e.
- `figuras/convergencia.png`: mejor MAE en CV por generación (mediana y RIC).
- `figuras/convergencia_arbol_vs_gep.png`: C con árboles frente a C con GEP.
- `figuras/originales_conservados.png`: fracción de ejecuciones que conserva cada original.
