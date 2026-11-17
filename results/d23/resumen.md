# D23: robustez a la partición en Diabetes (posterior a la validación)

Generado por `scripts/run_d23.py`. Descriptivo: con 5 particiones no se hacen tests. Cada partición usa las semillas 0–9; la 2026 es la de los experimentos finales (semillas 0–9 de `results/runs`).

| partición | n | e (validación) | MAE CV baseline (train) | e / CV | mediana e′ | % e′ < e | mejora rel. mediana | mejora (puntos) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 (referencia) | 10 | 39.22 | 46.28 | 0.847 | 38.6 | 80 | 1.59 % | +0.624 |
| 2027 | 10 | 40.53 | 45.5 | 0.891 | 40.82 | 30 | -0.73 % | -0.296 |
| 2028 | 10 | 45.85 | 44.38 | 1.033 | 46.13 | 40 | -0.61 % | -0.281 |
| 2029 | 10 | 44.37 | 44.48 | 0.998 | 45.48 | 10 | -2.48 % | -1.103 |
| 2030 | 10 | 41.06 | 45.55 | 0.902 | 41.54 | 20 | -1.17 % | -0.480 |
| 2031 | 10 | 40.91 | 45.7 | 0.895 | 39.53 | 100 | 3.35 % | +1.371 |

## Resumen entre particiones

- Particiones nuevas (5): la mediana de e′ es menor que e en **1 de 5**; mejora relativa mediana entre particiones -0.73 % (rango -2.48 % a 3.35 %).
- Incluida la 2026 (6): e′ < e en 2 de 6; mediana -0.67 % (rango -2.48 % a 3.35 %).

## Estabilidad de las fórmulas de C con árboles

### Partición 2026 (referencia)

Fórmulas más frecuentes:

- (2×) `sqrt(|(s6 + bmi)|)`
- (1×) `(((s5 / s3) * (s6 - age)) * sqrt(|s2|))`
- (1×) `(sex * age)`
- (1×) `(sqrt(|s5|) * s5)`
- (1×) `sqrt(|((bmi + sex) - (s4 - s2))|)`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 0.70, sex 0.60, bmi 0.70, bp 0.00, s1 0.10, s2 0.20, s3 0.40, s4 0.20, s5 0.40, s6 0.80

Fórmulas clave: log(1 + |s6|): 1/10 con la fórmula exacta, 1/10 con una fórmula que la contiene; sex·age: 1/10 con la fórmula exacta, 2/10 con una fórmula que la contiene

### Partición 2027

Fórmulas más frecuentes:

- (4×) `(bp * bmi)`
- (2×) `sqrt(|age|)`
- (2×) `(bmi * bp)`
- (2×) `log(1 + |s6|)`
- (1×) `log(1 + |log(1 + |s6|)|)`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 0.70, sex 0.10, bmi 0.80, bp 0.70, s1 0.00, s2 0.10, s3 0.20, s4 0.00, s5 0.20, s6 0.60

Fórmulas clave: log(1 + |s6|): 2/10 con la fórmula exacta, 3/10 con una fórmula que la contiene; sex·age: 1/10 con la fórmula exacta, 1/10 con una fórmula que la contiene

### Partición 2028

Fórmulas más frecuentes:

- (3×) `(sex * age)`
- (1×) `sqrt(|((sex + age) * log(1 + |age|))|)`
- (1×) `((bmi + (bmi + bmi)) * (bp - age))`
- (1×) `log(1 + |log(1 + |(s1 + bmi)|)|)`
- (1×) `(sqrt(|bmi|) + (sex * age))`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 0.90, sex 1.00, bmi 0.80, bp 0.60, s1 0.30, s2 0.30, s3 0.10, s4 0.20, s5 0.10, s6 0.20

Fórmulas clave: log(1 + |s6|): 0/10 con la fórmula exacta, 0/10 con una fórmula que la contiene; sex·age: 3/10 con la fórmula exacta, 5/10 con una fórmula que la contiene

### Partición 2029

Fórmulas más frecuentes:

- (2×) `(bp * bmi)`
- (2×) `(sex / s3)`
- (1×) `((bp * bmi) - (sex + s2))`
- (1×) `log(1 + |(age + s6)|)`
- (1×) `sqrt(|age|)`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 0.90, sex 0.60, bmi 0.90, bp 0.90, s1 0.20, s2 0.40, s3 0.60, s4 0.00, s5 0.40, s6 0.60

Fórmulas clave: log(1 + |s6|): 0/10 con la fórmula exacta, 1/10 con una fórmula que la contiene; sex·age: 0/10 con la fórmula exacta, 0/10 con una fórmula que la contiene

### Partición 2030

Fórmulas más frecuentes:

- (3×) `(sex * age)`
- (2×) `(age * sex)`
- (2×) `log(1 + |s6|)`
- (1×) `log(1 + |(s1 + (bmi - s3))|)`
- (1×) `sqrt(|(bp / age)|)`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 0.90, sex 1.00, bmi 0.70, bp 0.60, s1 0.60, s2 0.20, s3 0.40, s4 0.20, s5 0.30, s6 0.60

Fórmulas clave: log(1 + |s6|): 2/10 con la fórmula exacta, 2/10 con una fórmula que la contiene; sex·age: 5/10 con la fórmula exacta, 5/10 con una fórmula que la contiene

### Partición 2031

Fórmulas más frecuentes:

- (3×) `log(1 + |s6|)`
- (2×) `(age * sex)`
- (1×) `(sqrt(|s4|) / (sex * s6))`
- (1×) `log(1 + |(s6 + (age + s6))|)`
- (1×) `log(1 + |s2|)`

Fracción de ejecuciones que usan cada original en sus fórmulas: age 1.00, sex 0.50, bmi 0.60, bp 0.40, s1 0.20, s2 0.60, s3 0.10, s4 0.70, s5 0.10, s6 0.80

Fórmulas clave: log(1 + |s6|): 3/10 con la fórmula exacta, 3/10 con una fórmula que la contiene; sex·age: 2/10 con la fórmula exacta, 2/10 con una fórmula que la contiene

