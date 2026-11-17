# Diagnóstico de la cola de C con árboles en California (posterior a la validación)

Generado por `scripts/diagnose_tail.py`. Solo diagnóstico: no se ha cambiado ningún parámetro ni el código del sistema.

## Las 5 peores y las 5 mejores ejecuciones de C con árboles, y las 3 peores de C con GEP

**Peores, C con árboles**

| semilla | e′ | MAE CV | e′ − CV | columnas | originales conservados | fórmulas | complejidad | última mejora (gen.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.6618 | 0.5165 | +0.1454 | 7 | HouseAge, AveRooms, AveBedrms, AveOccup, Latitude | `(AveOccup * (Longitude * AveOccup))`; `(MedInc - Longitude)` | 15 | 100 |
| 28 | 0.5192 | 0.5310 | -0.0118 | 7 | MedInc, HouseAge, AveRooms, AveBedrms, Latitude, Longitude | `log(1 + |log(1 + |AveRooms|)|)` | 10 | 19 |
| 3 | 0.5165 | 0.5288 | -0.0123 | 7 | MedInc, HouseAge, AveRooms, AveBedrms, Longitude | `sqrt(|(Latitude * sqrt(|AveRooms|))|)`; `(Latitude + AveBedrms)` | 15 | 49 |
| 27 | 0.5133 | 0.5252 | -0.0120 | 7 | AveRooms, AveBedrms, Latitude, Longitude | `((AveRooms - HouseAge) + (AveOccup + AveBedrms))`; `(log(1 + |log(1 + |AveRooms|)|) + MedInc)`; `(log(1 + |Latitude|) * Latitude)` | 23 | 42 |
| 1 | 0.5116 | 0.5252 | -0.0136 | 7 | HouseAge, AveRooms, AveBedrms, AveOccup | `(Longitude - MedInc)`; `(sqrt(|sqrt(|Latitude|)|) - Latitude)`; `log(1 + |AveRooms|)` | 17 | 51 |

**Mejores, C con árboles**

| semilla | e′ | MAE CV | e′ − CV | columnas | originales conservados | fórmulas | complejidad | última mejora (gen.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.4698 | 0.4760 | -0.0062 | 7 | MedInc, HouseAge, Latitude, Longitude | `log(1 + |AveRooms|)`; `log(1 + |(Latitude * (Latitude + Longitude))|)`; `(AveOccup / sqrt(|AveOccup|))` | 19 | 70 |
| 0 | 0.4728 | 0.4780 | -0.0052 | 7 | MedInc, HouseAge, AveOccup, Longitude | `(sqrt(|(AveOccup * AveOccup)|) + AveOccup)`; `(Latitude - log(1 + |log(1 + |AveRooms|)|))`; `(log(1 + |Latitude|) - (sqrt(|Latitude|) - Longitude))` | 25 | 99 |
| 16 | 0.4734 | 0.4801 | -0.0067 | 7 | MedInc, HouseAge, Latitude, Longitude | `log(1 + |AveBedrms|)`; `(AveOccup / (AveOccup + sqrt(|AveOccup|)))`; `log(1 + |AveRooms|)` | 17 | 87 |
| 11 | 0.4734 | 0.4795 | -0.0061 | 7 | MedInc, HouseAge, Latitude, Longitude | `sqrt(|AveOccup|)`; `(log(1 + |log(1 + |AveRooms|)|) - (AveOccup / AveOccup))`; `sqrt(|(sqrt(|AveOccup|) + (AveOccup + AveOccup))|)` | 23 | 77 |
| 14 | 0.4738 | 0.4794 | -0.0056 | 7 | MedInc, AveOccup, Latitude, Longitude | `((Population + HouseAge) + (HouseAge + HouseAge))`; `(sqrt(|AveOccup|) * sqrt(|AveOccup|))`; `(sqrt(|log(1 + |Latitude|)|) + (sqrt(|AveRooms|) - Latitude))` | 27 | 84 |

**Peores, C con GEP (comparación)**

| semilla | e′ | MAE CV | e′ − CV | columnas | originales conservados | fórmulas | complejidad | última mejora (gen.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 0.5417 | 0.5189 | +0.0229 | 7 | MedInc, HouseAge, AveOccup, Latitude, Longitude | `((((Population + AveOccup) + Population) * AveOccup) - Latitude)`; `(sqrt(|AveRooms|) + sqrt(|AveOccup|))` | 21 | 91 |
| 26 | 0.4966 | 0.5047 | -0.0081 | 6 | MedInc, AveOccup, Latitude, Longitude | `(sqrt(|(AveOccup * (AveOccup - Latitude))|) + Latitude)`; `(HouseAge + AveBedrms)` | 17 | 98 |
| 16 | 0.4885 | 0.4979 | -0.0094 | 7 | HouseAge, AveRooms, AveBedrms, Latitude, Longitude | `((AveOccup * (MedInc / MedInc)) / log(1 + |log(1 + |AveOccup|)|))`; `(log(1 + |(AveOccup + AveRooms)|) + MedInc)` | 22 | 92 |

La complejidad es la del individuo ganador (columnas + nodos). «Última mejora» es la última generación en que mejoró el mejor fitness de la evolución (100 = seguía mejorando al terminar).

## ¿Mal CV (fallo de búsqueda) o buen CV y mal e′ (no generaliza)?

- Correlación de Spearman entre e′ y el MAE en CV del ganador: árboles ρ = 0.972 (p = 4.7e-19); GEP ρ = 0.939 (p = 1.7e-14).
- Las ejecuciones se separan en dos grupos por el mayor hueco en e′ (corte 0.4906): 11 malas y 19 buenas (37 % malas).
- MAE en CV mediano: malas 0.5165 frente a buenas 0.4838; e′ mediano: malas 0.5061 frente a buenas 0.4757; e′ − CV mediano: malas -0.0101, buenas -0.0061.
- Es decir, el e′ malo ya se ve en la CV sobre train: es un fallo de búsqueda (la evolución no encontró las transformaciones útiles), no de generalización.
- Excepciones, con e′ − CV > 0.02 (buen CV relativo y mal e′): árboles semilla 17 (e′ = 0.6618, CV = 0.5165); GEP semilla 27 (e′ = 0.5417, CV = 0.5189). En todas ellas hay una fórmula con un producto que incluye AveOccup (un término sin acotar); como causa es una hipótesis no verificada: comprobarla exigiría leer la validación.

## ¿Qué distingue a las malas de las buenas?

| medida | malas | buenas |
| --- | --- | --- |
| alguna fórmula transforma AveOccup (sqrt, log o cociente) | 45 % | 100 % |
| alguna fórmula contiene log(1 + \|AveRooms\|) | 27 % | 63 % |
| conservan MedInc como original | 73 % | 100 % |
| complejidad mediana | 17 | 19 |
| columnas medianas | 7 | 7 |
| última generación con mejora (mediana) | 74 | 86 |
| seguían mejorando al terminar (≥ gen. 95) | 18 % | 26 % |

La heurística de «transforma AveOccup» mira si una misma fórmula contiene AveOccup y sqrt, log o un cociente; las fórmulas completas están en las tablas de arriba y en los JSON.

## Hall of fame de las ejecuciones malas

| semilla | CV evolución del ganador | mejor CV evolución del hall of fame | CV folds nuevos del ganador | mejor CV folds nuevos del hall of fame |
| --- | --- | --- | --- | --- |
| 1 | 0.5259 | 0.5257 | 0.5252 | 0.5252 |
| 3 | 0.5286 | 0.5286 | 0.5288 | 0.5288 |
| 7 | 0.5149 | 0.5149 | 0.5154 | 0.5154 |
| 9 | 0.4967 | 0.4967 | 0.4967 | 0.4967 |
| 12 | 0.5177 | 0.5177 | 0.5174 | 0.5174 |
| 17 | 0.5164 | 0.5164 | 0.5165 | 0.5165 |
| 19 | 0.5127 | 0.5127 | 0.5124 | 0.5124 |
| 21 | 0.5051 | 0.5051 | 0.5051 | 0.5051 |
| 25 | 0.4944 | 0.4944 | 0.4952 | 0.4952 |
| 27 | 0.5251 | 0.5251 | 0.5252 | 0.5252 |
| 28 | 0.5311 | 0.5311 | 0.5310 | 0.5310 |

Ningún alternativo del hall of fame es sensiblemente mejor en CV: la mayor ventaja de un alternativo en la CV de la evolución es 0.0002. (El JSON guarda las métricas de los alternativos, no sus fórmulas; su e′ no se ha calculado para no seleccionar con validación.)

## Velocidad de la búsqueda

- Ejecuciones cuyo mejor MAE en CV baja de 0.5: árboles 21/30 (generación mediana 39.0), GEP 28/30 (generación mediana 29.0).
- RIC de e′: árboles 0.0277, GEP 0.0074.

## Una asimetría entre las dos representaciones (hipótesis de causa, no probada)

Con las tasas de los TOML, el número esperado de operaciones de mutación sobre fórmulas por individuo es 0.20 en árboles (`p_mutation` = 0,2: se muta un árbol en el 20 % de los individuos) frente a 2.58 en GEP (K genes de 15 símbolos con mutación puntual 0,044 por símbolo, más IS y RIS), es decir, unas 13 veces más. Es una consecuencia de usar «cada representación con sus tasas» (hito 4/6).
