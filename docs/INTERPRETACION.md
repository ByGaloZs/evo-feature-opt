# Interpretación de los resultados

Interpretación manual, posterior a la validación (2026-11-17). Es distinta de `results/analysis/informe.md`, que se genera automáticamente. No se ha cambiado ningún parámetro ni el código del sistema.

## H2 — «En Diabetes, la mejora es pequeña o nula»

**Criterio.** El diseño (D21) no fijó un umbral para «pequeña»; el criterio de abajo se decide después de ver los resultados de validación. Se compara la mejora mediana de C con árboles, en unidades de MAE, con (a) ε = 1,26, la escala del ruido de una estimación de CV en Diabetes, y (b) la mejora de California. Después se integra D23 (robustez a la partición).

**Partición 2026 (30 semillas, e = 39,22).**
- Mediana de e′ = 38,72: mejora de 0,503 puntos (1,28 %); e′ < e en el 73 % de las semillas; Wilcoxon unilateral p = 0,0053.
- Frente a ε: 0,503 / 1,26 = 0,40 ε. En California la mejora es de 0,0413 puntos = 13,8 ε (ε = 0,0030), y 7,92 % frente a 1,28 % (el 16 % de la de California).
- Con este criterio, la mejora en Diabetes es pequeña: menor que la escala del ruido de la CV.

**D23 (5 particiones nuevas, semillas 0–9 por partición).**

| Partición | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 |
| --- | --- | --- | --- | --- | --- | --- |
| mejora relativa mediana | +1,59 % | −0,73 % | −0,61 % | −2,48 % | −1,17 % | +3,35 % |
| mejora (puntos) | +0,62 | −0,30 | −0,28 | −1,10 | −0,48 | +1,37 |

- En las 5 nuevas, la mediana de e′ es menor que e en 1 de 5; mejora relativa mediana entre particiones −0,73 % (rango −2,48 % a +3,35 %). Con la 2026: 2 de 6, mediana −0,67 %.
- La diferencia es menor que ε en 5 de 6 particiones (solo la 2031, +1,37, la supera).
- Caso que corresponde: *mejora en unas particiones y empeora en otras, con mediana cercana a 0* → **no hay mejora fiable**.

**Conclusión.** H2 se apoya en el sentido de **nula**: la mejora de 1,28 % de la partición 2026 es estadísticamente significativa dentro de esa partición, pero no se reproduce en otras particiones; su signo cambia entre particiones y la mediana entre ellas es −0,73 %. El Wilcoxon solo recoge la variabilidad entre semillas con la partición fija, no entre particiones. La 2026 es además la partición con la validación más fácil frente a su CV (e / CV = 0,847; las otras, de 0,891 a 1,033).

**Límites.** Con 10 semillas por partición cada mediana es ruidosa. K, profundidad y ε se eligieron con el train de la 2026, que se solapa con la validación de las nuevas particiones (en Diabetes el piloto no distinguía entre combinaciones). D23 es descriptivo; con 5 particiones no se hacen tests.
