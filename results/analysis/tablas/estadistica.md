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
