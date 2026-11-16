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

