# Diseño y requisitos: optimización evolutiva de atributos

Nov 16, 2026 · @Mario Padilla

Propuesta: un transformador evolutivo P que sintetiza atributos con PG multiárbol (K árboles, uno por atributo nuevo) y selecciona originales con una máscara binaria, optimizando el MAE en validación cruzada sobre train sin tocar nunca validación. Como experimento adicional, se compara la representación en árbol con GEP manteniendo todo lo demás igual (D25). Cada decisión lleva su porqué y las alternativas descartadas; las marcadas como **Pendiente** las confirmas tú antes de escribir código.

## 1. Objetivo y criterio de éxito

La práctica cumple si un mismo modelo M, entrenado con los atributos transformados, consigue en validación un MAE menor que entrenado con los originales.

- **Entrada:** conjunto de entrenamiento (Xt, yt) y de validación (Xv, yv), todo valores reales (regresión).
- **P (nuestro programa):** aprende una transformación usando solo (Xt, yt) y la aplica igual a ambos conjuntos: Xt′ = P(Xt), Xv′ = P(Xv).
- **Línea base:** e = MAE(yv, M(Xv)), con M entrenado sobre (Xt, yt).
- **Resultado:** e′ = MAE(yv, M′(Xv′)), con M′ el mismo tipo de modelo e hiperparámetros, entrenado sobre (Xt′, yt).
- **Éxito:** e′ < e.

P puede hacer dos cosas, y nuestro diseño hace ambas: **síntesis** (nuevos atributos zi = f(x1…xn)) y **reducción de dimensionalidad** por selección de atributos.

```latex
e' = \frac{1}{m_v}\sum_{i=1}^{m_v} \left| y_{v,i} - M'(P(x_{v,i})) \right| \;<\; e = \frac{1}{m_v}\sum_{i=1}^{m_v} \left| y_{v,i} - M(x_{v,i}) \right|
```

## 2. Interpretación del enunciado

El enunciado deja ocho cosas abiertas; esta tabla fija cómo las leemos para que todo lo demás se apoye en ella.

| # | Ambigüedad | Decisión | Por qué |
| --- | --- | --- | --- |
| I1 | «Xy» y «v(Xv)» en las diapositivas | Son erratas de Xt y P(Xv) | El resto del texto define una sola P aplicada a ambos conjuntos |
| I2 | «Error absoluto» escrito como \|yv − M(Xv)\| (un vector) | Usamos el MAE (media) | El texto dice «error medio absoluto»; un vector no se puede comparar con < |
| I3 | «Modelo M arbitrario» | Elegimos un M principal (§5) y comprobamos la mejora en otros modelos como análisis secundario | Demostrar e′ < e para todo modelo es imposible; fijar uno es el enfoque *wrapper* estándar, y probar otros muestra si la mejora generaliza |
| I4 | Qué se entiende por «el mismo modelo» | Mismo algoritmo e hiperparámetros fijos, reentrenado sobre Xt′ | Así la única diferencia entre e y e′ son los datos, que es lo que se evalúa |
| I5 | Uso del conjunto de validación | Solo se toca una vez, al final, para medir e y e′ | Si guía la evolución, e′ deja de ser una estimación honesta (§4) |
| I6 | Si la partición train/validación viene dada | Si los CSV no traen partición, la hacemos una vez, fija y con semilla (§3) | Una partición única y documentada hace e comparable entre todos los experimentos |
| I7 | P es estocástico y el enunciado habla de un solo e′ | Varias ejecuciones con semillas distintas; se informa la mediana y el % de ejecuciones con e′ < e | Informar solo la mejor semilla sería elegir el resultado a dedo |
| I8 | Solo se entregan póster y memoria | Aun así, repositorio reproducible (RNF) | Los resultados del póster tienen que poder regenerarse si te los piden en la defensa |

## 3. Datos y preprocesado

Trabajamos con Diabetes.csv y California.csv; el primer paso del código es un análisis exploratorio que confirme las cifras de esta tabla.

| Dataset | Filas | Atributos | Objetivo | Rasgo a tener en cuenta |
| --- | --- | --- | --- | --- |
| Diabetes | ≈ 442 | 10 | Progresión de la enfermedad al año | Pocas filas: la validación tendrá ≈ 90 y e será ruidoso |
| California | ≈ 20 640 | 8 | Valor mediano de la vivienda | Objetivo recortado en un máximo; relaciones claramente no lineales (lat/long, renta) |

Cifras aproximadas de las versiones públicas de scikit-learn; **a verificar con tus CSV**.

**D1 — Partición train/validación.** Si los CSV no traen partición: 80/20 aleatoria, semilla fija, hecha una vez y guardada en disco.

- *Por qué:* 80/20 deja suficiente train para que la evolución aprenda y una validación usable incluso en Diabetes. Guardarla en disco garantiza que todos los experimentos miden contra el mismo e.
- *Descartado:* 70/30 (quita train a un dataset ya pequeño); CV anidada como evaluación final (más robusta, pero el enunciado define un único e′ sobre un conjunto de validación).

**D2 — Escalado.** Estandarización (media 0, desviación 1) ajustada solo con train, en dos puntos: sobre los originales antes de la síntesis y sobre la matriz final Xt′.

- *Por qué antes:* con escalas comparables, las funciones (×, √, log) generan atributos de magnitud razonable y las protecciones numéricas actúan menos.
- *Por qué después:* modelos como KNN o Ridge dependen de la escala; para regresión lineal es inocuo.
- *Por qué solo con train:* ajustarlo con validación filtraría información (§4).

**D3 — Faltantes.** El enunciado dice que todo es real; si el análisis encuentra huecos, imputación por mediana ajustada en train. *Por qué:* la mediana es robusta a atípicos y no inventa estructura.

**D4 — La variable objetivo no se transforma.** *Por qué:* el criterio es el MAE sobre y original, y el enunciado pide optimizar la matriz de atributos, no el objetivo.

**D5 — No se eliminan filas (instancias)** (atípicos, duplicados, valores recortados). *Por qué:* limpiar filas cambia el problema y mezclaría el efecto de la limpieza con el de P; se menciona como limitación en la memoria.

D5 se refiere solo a filas. La reducción de dimensionalidad quita **columnas** (atributos) y la hace la máscara de selección (D13).

## 4. Protocolo de evaluación y prevención de fugas

La regla central: la evolución solo ve train, y validación se lee una vez, en el script de evaluación final.

**D6 — Fitness por validación cruzada interna.** Cada individuo se evalúa con 5-fold CV sobre (Xt, yt). Las particiones (folds) se fijan al inicio y son las mismas para todos los individuos de una ejecución.

- *Por qué CV:* el error en train premia el sobreajuste (más atributos siempre bajan el error de entrenamiento); la CV estima el error en datos no vistos.
- *Por qué k = 5:* compromiso habitual entre sesgo, varianza y coste; en Diabetes cada fold de test tiene ≈ 70 filas, menos sería demasiado ruidoso.
- *Por qué folds fijos:* todos los individuos se comparan en igualdad de condiciones y el fitness no tiene ruido entre generaciones.
- *Descartado:* un hold-out simple dentro de train (más barato, pero mucho más ruidoso con 350 filas); folds nuevos en cada generación (reduce el sobreajuste a los folds, pero añade ruido a la selección; D7 ataca ese problema de forma más barata).

**D7 — Selección final con folds nuevos.** Al terminar, los 10 mejores individuos vistos (*hall of fame*) se re-evalúan con CV repetida (3 × 5-fold) sobre particiones distintas de las de la evolución, y gana el mejor ahí.

- *Por qué:* tras miles de evaluaciones sobre los mismos folds, el mejor individuo está sesgado a favor de esos folds. Re-evaluar con particiones nuevas corrige ese sesgo de selección sin tocar validación.

**D8 — Todo lo que se ajusta, se ajusta dentro de train.** Imputación, escalado y modelo se ajustan solo con los datos de entrenamiento de cada momento: con Xt en P.fit, y con la parte de entrenamiento de cada fold dentro de la CV (un pipeline escalado + modelo por fold).

- *Por qué:* ajustar un escalador con datos de test, aunque sea dentro de un fold, filtra su media y varianza y hace la CV optimista.

**D9 — Interfaz fit / transform** al estilo scikit-learn: `P.fit(Xt, yt)` aprende y `P.transform(X)` solo aplica.

- *Por qué:* separar aprender de aplicar hace la fuga imposible por construcción, y P encaja en un `Pipeline` de scikit-learn sin código especial.

## 4.1 Arquitectura

```text
Train (Xt, yt) ──► P.fit (solo train)
                    ├─ escalado de originales (ajustado en train)
                    └─ bucle evolutivo, 100 generaciones:
                         Población (máscara + K árboles)
                           → Evaluar fórmulas → Xt′ candidata (s < n)
                           → Fitness: MAE en 5-fold CV con M
                           → Torneo + operadores → (siguiente generación)
                  ──► Re-selección: top 10, CV repetida en folds nuevos
                  ──► P final (máscara + fórmulas fijadas)

Validación (Xv, yv) ── sin uso hasta aquí ──► P.transform (aplica sin reaprender)
                                            ──► ¿e′ < e? (mismo M sobre X y sobre X′)
```

Todo lo que P aprende (escalado, fórmulas, máscara) sale de train; validación solo entra en las dos últimas cajas.

## 5. Modelo de referencia M

Propuesta: regresión lineal ordinaria como M principal (el que guía el fitness), y otros tres modelos solo para comprobar si la mejora generaliza.

**D10 — M principal: regresión lineal (OLS).** **Decidido.**

- *Por qué:* es el caso que el propio enunciado pone de ejemplo (síntesis no lineal + modelo lineal), así que es donde P tiene más margen y donde mejor se explica el efecto.
- *Por qué:* no tiene hiperparámetros y es determinista, así que e es un número fijo y ningún ajuste de hiperparámetros contamina la comparación.
- *Por qué:* entrena en milisegundos incluso con 16 000 filas, lo que permite decenas de miles de evaluaciones de fitness.
- *Por qué:* sus coeficientes sobre los atributos sintetizados se pueden leer, lo que da material para el análisis.
- *Crítica honesta:* elegir un modelo débil hace la mejora más fácil. Lo compensamos con D11 y lo decimos en la memoria.
- *Alternativa seria:* Ridge con α fijo pequeño. Tolera mejor atributos sintetizados muy correlacionados entre sí, a cambio de un hiperparámetro que hay que justificar.
- *Descartado como principal:* Random Forest o gradient boosting (ya capturan no linealidades, así que el margen es pequeño, y son 100–1000 veces más lentos dentro del fitness); KNN (depende mucho del valor de k).

**D11 — Modelos secundarios, solo en la evaluación final:** Ridge, KNN y Random Forest, con hiperparámetros fijos y documentados.

- *Por qué:* el enunciado habla de un M «arbitrario» (I3). Medir e y e′ también con estos modelos dice si la transformación es buena en general o solo para OLS.
- *Qué esperamos* (hipótesis que se escribe antes de ejecutar): mejora clara en Ridge, posible mejora en KNN gracias a la selección, poca o ninguna en Random Forest.

## 6. Representación del individuo

Un único individuo codifica a la vez qué originales se quedan (máscara binaria, como un AG) y qué atributos nuevos se crean (K árboles de PG, uno por atributo). Es el esquema de M3GP adaptado a selección + síntesis.

**D12 — Técnica: PG multiárbol + máscara de selección, en un solo individuo.** Decidido; GEP queda como experimento de comparación (D25).

- *Por qué síntesis y no solo selección:* con OLS en California los 8 atributos aportan; quitar columnas apenas mejora. La mejora tiene que venir de crear relaciones no lineales, que un AG de selección no puede generar.
- *Por qué árboles:* es la representación de toda la literatura de construcción de atributos con PG (§14). Nuestro diseño es una variante reconocible de M3GP, que hace justo esto en regresión: transforma los datos con varios árboles y ajusta un modelo lineal encima.
- *Por qué un solo individuo y no dos etapas* (primero sintetizar, luego seleccionar): lo que vale un atributo nuevo depende de qué originales se quedan, y al revés. Dos etapas optimizan cada parte por separado y pierden esas interacciones. La ablación del §9 sigue siendo posible desactivando una de las partes.
- *Por qué GEP pasa a comparación:* sus ventajas (cromosoma lineal, siempre válido, tamaño acotado) son de ingeniería; no hay evidencia independiente de que dé mejores resultados ni precedentes en construcción de atributos. Compararlo con todo lo demás igual convierte esa duda en una pregunta experimental (D25).
- *Descartado:* EE y DE como técnica principal, porque optimizan vectores de reales y no representan la estructura de una fórmula. Quedan como extensión para ajustar constantes (D15).

**D13 — Estructura del individuo.**

| Parte | Tamaño | Qué codifica |
| --- | --- | --- |
| Máscara de originales | n bits (10 en Diabetes, 8 en California) | Qué atributos originales pasan a Xt′ |
| Árboles | K árboles, profundidad máxima 3 (≤ 15 nodos cada uno) | Un atributo sintetizado por árbol |
| Interruptores de árboles | K bits | Qué atributos sintetizados pasan a Xt′ |

- Valores de partida: **K = 5, profundidad máxima 3** (la raíz es profundidad 0). Los definitivos se eligen en el piloto (D20) con una rejilla K ∈ {3, 5, 7} × profundidad ∈ {2, 3, 4}.
- *Por qué profundidad 3 de partida:* con operaciones binarias da como mucho 15 nodos. Profundidad 2 (7 nodos) no admite una transformación de una interacción dividida por otro atributo, como log(1+|x₁·x₂|)/x₃, que es el tipo de atributo que esperamos útil. Profundidad 4 (31 nodos) o más multiplica el espacio de búsqueda, aumenta el riesgo de sobreajuste con las ≈ 350 filas de train de Diabetes y produce fórmulas que no se pueden explicar. La profundidad 17 de Koza es para PG general, no para construir atributos.
- *Por qué K = 5 de partida:* con la restricción de D24, Xt′ tiene como mucho n − 1 columnas (7 en California, 9 en Diabetes), así que K > n − 1 nunca se podría usar entero. K = 5 deja sitio para 5 atributos nuevos y al menos 2 originales en California. Más árboles agrandan el individuo y añaden colinealidad; menos impedirían usar más atributos nuevos aunque convenga. Con los interruptores la evolución puede usar menos de K, así que es preferible que K sobre a que falte.
- *Por qué interruptores por árbol:* el número de atributos nuevos también se optimiza, entre 0 y K, en vez de forzar siempre K.
- Los árboles pueden usar **todos** los originales, aunque la máscara los quite de Xt′. *Por qué:* un atributo puede ser inútil solo y útil dentro de un producto.
- Si un individuo queda sin ningún atributo, se repara activando un bit al azar. *Por qué reparar y no penalizar:* no desperdicia evaluaciones en individuos imposibles.

**D24 — Xt′ con menos columnas que X.** **Decidido**: restricción dura, columnas de Xt′ < n.

- *Por qué:* el enunciado define la reducción de dimensionalidad como pasar a (z₁, …, zₛ) con s < n, donde los z pueden ser atributos nuevos; la selección es solo el caso particular de un subconjunto. Una Xt′ con menos columnas, mezclando originales y sintetizados, cumple las dos definiciones del enunciado a la vez, y cumple con cualquier lectura de lo que espera el profesor.
- *Coste esperado:* algo menos de mejora en California, donde los 8 originales aportan. La literatura sugiere que es asumible: pocos atributos compactos construidos con PG pueden competir con el conjunto original completo (Virgolin et al., §14).
- *Cómo se aplica:* si un individuo supera el límite, se desactivan columnas al azar hasta cumplirlo. *Por qué reparar:* igual que en D13, no se desperdician evaluaciones.
- *Alcance:* la restricción se aplica a la configuración C (el sistema propuesto) y a su variante GEP. La configuración B del §9 (solo síntesis, todos los originales) la incumple por definición; es un diagnóstico de la ablación, no una solución.
- *Alternativas:* sin restricción (más margen de mejora, pero solo es válida con una de las dos lecturas del enunciado) o preguntar al profesor.

**D14 — Conjunto de funciones:** +, −, ×, ÷ protegida, Q(x) = √|x|, L(x) = log(1 + |x|).

- *Por qué +, −, ×:* las interacciones entre pares (xᵢ·xⱼ) son el atributo sintetizado útil más frecuente.
- *Por qué ÷:* captura cocientes (por ejemplo, población por vivienda en California).
- *Por qué Q y L:* comprimen distribuciones asimétricas (renta, población), que es la no linealidad típica de estos datos.
- *Por qué protegidas:* ÷ devuelve 1 si |denominador| < 10⁻⁶; Q y L usan |x|. Así la fórmula da un valor finito en cualquier dato, también en validación, que la evolución nunca vio.
- *Descartado:* exp (desbordamientos), funciones trigonométricas (sin sentido físico aquí y más espacio de búsqueda), if-then-else (fórmulas difíciles de interpretar).
- Si un gen da valores no finitos o constantes en train, se desactiva. *Por qué:* una columna constante no aporta nada y una no finita rompe el modelo.

**D15 — Terminales: los originales estandarizados, sin constantes numéricas.**

- *Por qué:* OLS ajusta un coeficiente y un término independiente por atributo, así que las constantes en el nivel exterior de la fórmula son redundantes. Nos ahorramos los mecanismos de constantes (constantes efímeras en PG, dominio Dc en GEP).
- *Limitación aceptada:* se pierden constantes internas (por ejemplo, log(1 + x₁ − 0,5)). Extensión posible: ajustarlas con DE.

## 7. Función de fitness y control de complejidad

El fitness es el MAE medio de M en la CV interna (a minimizar); entre individuos casi empatados gana el más simple.

**D16 — Fitness = MAE medio en 5-fold CV sobre train.**

- *Por qué MAE:* es exactamente la métrica del criterio de éxito. Optimizar el MSE optimizaría otra cosa.
- *Matiz:* OLS minimiza internamente el error cuadrático; aceptamos ese desajuste porque lo que se elige con MAE es la transformación, no los coeficientes.

**D17 — Parsimonia lexicográfica.** En cada torneo, si dos individuos difieren en fitness menos de un umbral ε, gana el de menor complejidad. Complejidad = número de columnas de Xt′ + nodos de los árboles (o genes) activos.

- *Por qué:* evita sobreajustar con atributos que bajan el error de forma insignificante, y da fórmulas más legibles para el póster.
- *Por qué lexicográfica y no una penalización MAE + λ·complejidad:* no obliga a mezclar unidades distintas ni a calibrar λ para cada dataset.
- *Descartado:* multiobjetivo con NSGA-II. Da un frente de Pareto error/complejidad muy vistoso, pero añade un algoritmo nuevo que no está en las presentaciones.
- Valor de ε: **Pendiente** del piloto. Propuesta: la desviación típica del MAE entre folds dividida por √5; diferencias menores que eso son ruido de la CV.

**D18 — Caché de fitness por fenotipo.** Antes de evaluar, se calcula una huella de las fórmulas expresadas y la máscara; si ya se evaluó, se reutiliza.

- *Por qué:* muchos individuos distintos expresan la misma fórmula (la selección duplica individuos y, en GEP, la parte final del gen no siempre se usa), así que la caché ahorra muchas evaluaciones y no cambia ningún resultado.

## 8. Algoritmo evolutivo

Algoritmo generacional con elitismo, selección por torneo y operadores de PG con árboles, con un presupuesto fijo de 100 × 100 evaluaciones. Todos los valores son de partida y se confirman en el piloto.

| Elemento | Valor propuesto | Por qué |
| --- | --- | --- |
| Esquema | Generacional, elitismo de 2 | El mejor individuo nunca se pierde, así que la curva del mejor fitness es monótona y se puede explicar |
| Población | 100 | Compromiso habitual entre diversidad y coste para individuos de este tamaño |
| Parada | 100 generaciones fijas | Mismo presupuesto en todas las ejecuciones, ablaciones y representaciones, así que son comparables. Parar por estancamiento haría que cada ejecución gaste distinto |
| Selección | Torneo de tamaño 3 | Funciona igual minimizando que maximizando y no depende de la escala del MAE (la ruleta sí); admite la parsimonia lexicográfica de D17 de forma natural |
| Inicialización de árboles | *Ramped half-and-half* (mitad Grow, mitad Full), profundidades 1 a 3 | Variedad de formas y tamaños desde el principio; es el método estándar de Koza y está en las diapositivas de PG |
| Inicialización de bits | Cada bit al 50 % | Diversidad máxima en qué atributos se usan |
| Individuos semilla | Uno por cada original eliminado: n individuos con n − 1 originales y ningún árbol activo | Con D24 la identidad (todos los originales) ya no es válida; estas semillas garantizan que la mejor selección de n − 1 atributos es alcanzable desde el principio |
| Cruce de árboles | Probabilidad 0,8 por pareja: cruce de subárbol entre los árboles de la misma posición k, o intercambio de árboles completos (50 % cada uno) | El de subárbol recombina partes de fórmulas; el intercambio pasa atributos completos que ya funcionan, como la recombinación de genes de GEP |
| Cruce de bits | Uniforme | El orden de los atributos es arbitrario, así que no hay bloques contiguos que preservar |
| Mutación de árboles | Probabilidad 0,2 por individuo: de subárbol (lo sustituye por uno nuevo Grow) o puntual (cambia un nodo por otro de igual aridad), 50 % cada una | La de subárbol explora fórmulas nuevas; la puntual hace ajustes finos sin cambiar la forma |
| Mutación de bits | 1/(n + K) por bit | Se cambia de media un bit por individuo: explora sin destruir |
| Límite de profundidad | Un hijo que supera la profundidad 3 se descarta y se conserva el padre | Regla clásica de Koza; mantiene el tamaño acotado sin reparar árboles |
| Semillas aleatorias | 30 por configuración y dataset | Suficiente para medianas estables y tests no paramétricos (§9) |

- *Coste estimado:* 10 000 evaluaciones × 5 folds de OLS por ejecución. En California son unos minutos por ejecución, menos con la caché de D18. Se confirma midiendo en el piloto antes de lanzar las 30 semillas.
- *Implementación:* todo propio con NumPy (árboles, GEP y bucle evolutivo); scikit-learn solo para los modelos, el escalado y las particiones de CV. **Decidido.**
  - *Por qué:* control total y cada pieza se puede explicar en la defensa. Además, GEP se decodifica a la misma representación de árbol que usa la PG (§15), así que evaluación, impresión de fórmulas y caché son el mismo código para ambas: la comparación de D25 es justa por construcción.
  - *Coste aceptado:* más código propio y más riesgo de errores sutiles. Se compensa con tests por hito (§15.6), incluido un problema sintético con solución conocida.
  - *Descartado:* el módulo de PG de DEAP (menos código, pero la comparación con un GEP propio mezclaría dos implementaciones distintas).
- *Variante GEP (solo para D25):* h = 7, así que t = 8 y cada gen tiene 15 símbolos, como mucho 15 nodos, igual que un árbol de profundidad 3. Operadores y tasas por defecto de GEP (Ferreira): mutación 0,044 por símbolo, transposiciones IS, RIS y de gen 0,1 cada una, recombinación de 1 punto, 2 puntos y de genes 0,3 / 0,3 / 0,1.

## 9. Diseño experimental

Cuatro configuraciones por dataset más la variante GEP de C, 30 semillas cada una, con hipótesis escritas antes de ejecutar y tests no paramétricos.

| Configuración | Máscara | Genes | Qué responde |
| --- | --- | --- | --- |
| Línea base | Todos los originales | Ninguno | El valor de e |
| A — Solo selección | Evoluciona | Desactivados | ¿Cuánto aporta quitar atributos? |
| B — Solo síntesis | Todos los originales, fija | Evolucionan | ¿Cuánto aporta crear atributos? |
| C — Completo | Evoluciona | Evolucionan | El sistema propuesto |

**D19 — Ablación A / B / C.** *Por qué:* si C mejora, la ablación dice de dónde viene la mejora. Sin ella, el análisis de la memoria se queda en «funciona» o «no funciona».

**D20 — Piloto antes de los experimentos.** Rejilla K ∈ {3, 5, 7} × profundidad máxima ∈ {2, 3, 4} (9 combinaciones), 3 semillas cada una, configuración C, usando **solo la CV sobre train**. Gana la combinación con mejor MAE en CV; si varias quedan dentro de ε (D17), la de menor K y profundidad. El piloto también fija ε y mide los tiempos.

- *Por qué una rejilla:* convierte K y la profundidad en valores elegidos con datos en lugar de por criterio, y en la memoria se puede decir cómo se eligieron.
- *Por qué solo con la CV sobre train:* ajustar parámetros mirando validación también es una fuga, aunque sea indirecta.
- *Por qué 3 semillas:* basta para descartar combinaciones claramente peores; las 30 semillas se reservan para los experimentos definitivos. Coste: 27 ejecuciones cortas por dataset.
- *Variante GEP:* h se deduce de la profundidad elegida para igualar el tamaño máximo (profundidad 2 → h = 3, 3 → h = 7, 4 → h = 15), así que no necesita piloto propio para el tamaño.

**D21 — Hipótesis escritas antes de ejecutar** (en un archivo del repositorio, con fecha):

- H1: en California, C mejora claramente, porque sus relaciones son no lineales.
- H2: en Diabetes, la mejora es pequeña o nula, porque la relación es casi lineal, hay pocas filas y la validación es ruidosa.
- H3: B aporta más que A en ambos datasets.
- *Por qué:* escribirlas antes impide reinterpretar los resultados a posteriori, y una hipótesis fallida con su explicación también es un buen resultado para la memoria.

**D22 — Métricas y estadística.**

- Por ejecución: e′, mejora relativa (e − e′)/e, número de columnas de Xt′, fórmulas elegidas, fitness en CV y tiempo.
- Por configuración: mediana y rango intercuartílico de e′, y % de semillas con e′ < e.
- ¿Mejora C sobre la línea base? Wilcoxon de una muestra sobre e′ − e (unilateral, α = 0,05). Con OLS, e es un valor fijo.
- ¿Difieren A, B y C? Kruskal-Wallis y, si hay diferencias, Mann-Whitney por pares con corrección de Holm.
- *Por qué no paramétricos:* con 30 ejecuciones no podemos suponer normalidad en e′.
- Diferencia entre el fitness en CV y e′: si es grande, indica sobreajuste a train.

**D23 — Robustez a la partición (opcional).** Repetir C con 5 particiones train/validación distintas, sobre todo en Diabetes. *Por qué:* con ≈ 90 filas de validación, una sola partición puede dar una mejora o un empeoramiento por suerte.

**D25 — Comparación de representaciones: árboles frente a GEP.** La configuración C se ejecuta dos veces, con árboles y con GEP, cambiando **solo** la representación de las fórmulas.

- *Qué se mantiene igual:* máscara, interruptores, fitness, parsimonia, conjunto de funciones, tamaño máximo de fórmula (15 nodos), presupuesto (100 × 100 evaluaciones), restricción de D24 y semillas.
- *Qué se compara:* distribución de e′ (Mann-Whitney), curvas de convergencia, tamaño de las fórmulas finales y tiempo.
- *H4 (escrita antes de ejecutar):* no habrá diferencia significativa en e′, porque con el mismo tamaño máximo ambas pueden expresar prácticamente las mismas fórmulas; si hay diferencias, aparecerán en la velocidad de convergencia.
- *Por qué:* GEP aporta originalidad, pero no hay evidencia independiente de que mejore a la PG con árboles en este uso. Con la comparación, cualquier resultado (gana uno, gana el otro o empatan) es un hallazgo para la memoria, en lugar de una apuesta.
- *Por qué solo en C:* es la configuración que importa y la comparación cuesta 30 ejecuciones más por dataset, no 90.

## 10. Requisitos

Cada requisito sale de una decisión anterior; si una decisión cambia, sabemos qué requisitos revisar.

**Funcionales**

| ID | Requisito | Viene de |
| --- | --- | --- |
| RF-01 | Cargar cada CSV, comprobar que todo es numérico, detectar faltantes y separar X de y | D3 |
| RF-02 | Crear la partición train/validación una vez, con semilla, y guardarla en disco; si ya existe, leerla | D1 |
| RF-03 | Exponer P como transformador con `fit(Xt, yt)` y `transform(X)` | D9 |
| RF-04 | Evaluar los árboles (y, en la variante GEP, decodificar los genes Karva) de forma vectorizada con NumPy, con operaciones protegidas | D14, D15 |
| RF-05 | Calcular el fitness con 5-fold CV de folds fijos y un pipeline escalado + M por fold | D6, D8, D16 |
| RF-06 | Implementar los operadores de árboles del §8 y, para la variante GEP, mutación, transposiciones y recombinaciones respetando cabeza/cola; en ambas, los operadores de máscara e interruptores, y la reparación de D24 | D13, §8 |
| RF-07 | Torneo con parsimonia lexicográfica y caché de fitness por fenotipo | D17, D18 |
| RF-08 | Guardar el hall of fame y re-seleccionar el ganador con CV repetida en folds nuevos | D7 |
| RF-09 | Calcular e y e′ con M y con los modelos secundarios | D10, D11 |
| RF-10 | Ejecutar las configuraciones línea base, A, B y C desde un mismo código cambiando solo la configuración | D19 |
| RF-11 | Registrar por ejecución: métricas, fórmulas en notación legible, curva de convergencia y tiempo | D22 |
| RF-12 | Script de análisis que genere las tablas, los tests estadísticos y las figuras a partir de los registros | D22, §11 |

**No funcionales**

| ID | Requisito | Por qué |
| --- | --- | --- |
| RNF-01 | Una semilla controla toda la aleatoriedad; cada experimento se define en un archivo de configuración; versiones de librerías fijadas | Cualquier resultado del póster se puede regenerar exactamente |
| RNF-02 | Solo el script de evaluación final lee el archivo de validación | La regla anti-fuga del §4 se cumple por cómo está organizado el código, no por disciplina |
| RNF-03 | Una ejecución en California cabe en el tiempo fijado tras el piloto | 30 semillas × 4 configuraciones × 2 datasets tienen que caber en una tarde |
| RNF-04 | Las fórmulas se imprimen en notación infija con los nombres reales de las columnas | Se pueden leer e interpretar en el póster |
| RNF-05 | Tests unitarios de la evaluación de árboles, de la decodificación Karva con los ejemplos de las diapositivas (+/Q\*c-abde, \*-/Qb+b+aaab) y de las operaciones protegidas | Es la pieza más fácil de implementar mal sin que se note |
| RNF-06 | Python 3 con NumPy, SciPy, scikit-learn y Matplotlib | Librerías estándar, sin dependencias exóticas |

## 11. Entregables

El póster explica la solución y los experimentos; la memoria contiene solo resultados y su análisis, como pide el enunciado. Este diseño tiene que producir todo lo de esta tabla.

| Pieza | Póster | Memoria | Sale de |
| --- | --- | --- | --- |
| Esquema del sistema (flujo fit/transform y bucle evolutivo) | Sí | No | §4, §6 |
| Ejemplo de individuo (máscara + árboles) y sus fórmulas; comparación árboles frente a GEP (D25) | Sí | No | D13 |
| Tabla e vs e′ (mediana, rango intercuartílico, % de éxito) por dataset y configuración | Resumida | Completa | D22 |
| Diagramas de caja de e′ por configuración, con línea en e | Sí | Sí | D19, D22 |
| Curvas de convergencia (mediana y banda entre semillas) | Sí | Sí | RF-11 |
| Atributos sintetizados más frecuentes y atributos más eliminados | Sí | Sí | RF-11 |
| Resultados con modelos secundarios | Resumida | Completa | D11 |
| Tests estadísticos y contraste de hipótesis H1–H3 | Conclusión | Completo | D21, D22 |

## 12. Riesgos

| Riesgo | Señal | Mitigación |
| --- | --- | --- |
| Sobreajuste a la CV de train | El fitness en CV mejora pero e′ no | D7 (re-selección con folds nuevos), D17 (parsimonia), fórmulas cortas (h = 4) |
| Ninguna mejora en Diabetes | e′ ≥ e en la mayoría de semillas | Está previsto en H2; se analiza por qué (relación casi lineal, pocas filas) en lugar de forzar la mejora |
| Validación de Diabetes demasiado pequeña | Resultados que cambian mucho con la partición | D23 (varias particiones) |
| Atributos sintetizados muy correlacionados | Coeficientes de OLS enormes o inestables | Interruptores por gen y parsimonia; alternativa Ridge (D10) |
| Coste de cómputo en California | El piloto supera el tiempo de RNF-03 | Caché (D18); bajar población o generaciones por igual en todas las configuraciones |
| Errores en la evaluación de árboles o en la decodificación Karva | Fórmulas que no coinciden con lo esperado | RNF-05 (tests con los ejemplos de clase) |
| Valores extremos en validación | Atributos sintetizados con valores muy fuera del rango de train | Operaciones protegidas (D14) y escalado ajustado en train; se revisa en el análisis |

## 13. Decisiones pendientes de tu confirmación

Nada de código hasta cerrar las cinco primeras; las dos últimas se cierran con el piloto.

- [ ] D1 — Confirmar si los CSV traen la partición hecha y qué columna es y (lo comprueba el hito 1 del §15.6)
- [ ] D23 — Incluir o no el estudio de robustez a la partición
- [ ] D13 — Valores finales de K y profundidad máxima (los elige el piloto, D20)
- [ ] D17 — Valor final de ε (lo fija el piloto, D20)

Ya decidido: D10 (OLS), D12 (PG multiárbol + máscara), D24 (Xt′ con menos columnas que X), D25 (comparación con GEP) e implementación propia con NumPy (§8).

## 14. Fundamentación

La literatura respalda la idea general del diseño: construir y seleccionar atributos a la vez con PG, con un modelo lineal encima y pocos atributos compactos. Los detalles concretos (máscara de bits, restricción s < n, comparación con GEP) son decisiones propias, justificadas en cada sección.

| Referencia | Qué aporta | Respalda |
| --- | --- | --- |
| [Muñoz, Trujillo, Silva, Castelli y Vanneschi (2019), *Evolving multidimensional transformations for symbolic regression with M3GP*, Memetic Computing 11(2)](https://gpbib.cs.ucl.ac.uk/gp-html/Munoz_2019_MemeticC.html) | PG multiárbol que transforma los datos a un espacio de dimensión elegida por la búsqueda y ajusta un modelo lineal por mínimos cuadrados; menos error y modelos más compactos que la mayoría de métodos comparados | D12, D13, D10 |
| [Tran, Xue y Zhang (2016), *Genetic programming for feature construction and selection in classification on high-dimensional data*, Memetic Computing 8(1)](https://gpbib.cs.ucl.ac.uk/gp-html/journals_memetic_TranXZ16.html) | PG para construir atributos y seleccionarlos implícitamente; reduce mucho la dimensionalidad manteniendo o mejorando la precisión | D12, D24 |
| [Virgolin, Alderliesten y Bosman (2020), *On explaining machine learning models by evolving crucial and compact features*, Swarm and Evolutionary Computation 53](https://repository.tudelft.nl/record/uuid:6a354e4b-5162-4da2-9f75-3ddc04035296) | En 21 datasets de clasificación y regresión, dos atributos compactos construidos con PG pueden competir con todo el conjunto original | D24, D13, D17 |
| [Krawiec (2002), *Genetic programming-based construction of features for machine learning and knowledge discovery tasks*, Genetic Programming and Evolvable Machines 3(4)](https://gpbib.cs.ucl.ac.uk/gp-html/krawiec_2002_GPEM.html) | Los atributos construidos con PG mejoran la precisión de los clasificadores | D12 |
| [Oltean y Grosan (2004), *A comparison of several linear genetic programming techniques*, Complex Systems 14(4)](https://gpbib.cs.ucl.ac.uk/gp-html/oltean_2004_CS.html) | Comparación independiente de GEP con otras variantes lineales de PG; las mejores fueron MEP y LGP | D25 |
| Kohavi y John (1997), *Wrappers for feature subset selection*, Artificial Intelligence 97 | Enfoque *wrapper*: el fitness es el error del propio modelo. Citado de memoria, sin verificar | D6, D16 |
| Diapositivas del curso (P. Isasi): PG, Gene Expression Programming, AG, Conceptos de CE | Inicialización Grow/Full, operadores de subárbol, estructura cabeza/cola y operadores de GEP, torneo y elitismo | §8, D25 |

## 15. Especificación de implementación

Esta sección es la que sigue quien escriba el código: estructura, interfaces, formatos, valores y orden de trabajo. Si algo no está aquí ni en las secciones anteriores, se pregunta antes de decidirlo.

### 15.1 Estructura del proyecto

```text
evo-feature-opt/
├── CLAUDE.md                # reglas para Claude Code
├── README.md                # cómo reproducir cada resultado
├── pyproject.toml           # Python ≥ 3.11 y dependencias con versión fijada
├── docs/
│   ├── DISENO.md            # este documento exportado
│   └── HIPOTESIS.md         # H1–H4, con fecha, antes de ejecutar (D21)
├── data/
│   ├── raw/                 # Diabetes.csv, California.csv (sin tocar)
│   └── splits/              # <dataset>_train.csv, <dataset>_val.csv (D1)
├── configs/                 # un .toml por experimento
├── src/evofeat/
│   ├── data.py
│   ├── primitives.py
│   ├── formula.py
│   ├── repr_tree.py
│   ├── repr_gep.py
│   ├── individual.py
│   ├── fitness.py
│   ├── evolution.py
│   ├── transformer.py
│   ├── models.py
│   └── experiment.py
├── scripts/                 # make_splits, run_pilot, run_experiments, analyze
├── results/                 # un JSON por ejecución + tablas y figuras
└── tests/
```

| Módulo | Responsabilidad | Requisitos |
| --- | --- | --- |
| data.py | Cargar y validar CSV, crear y leer particiones. `load_validation()` vive aquí y solo la llama experiment.py | RF-01, RF-02, RNF-02 |
| primitives.py | Funciones protegidas (+, −, ×, ÷, Q, L) con su aridad | D14 |
| formula.py | Representación común de una fórmula: tupla de símbolos en preorden. Evaluación vectorizada, profundidad, tamaño, impresión infija y huella | RF-04, RNF-04 |
| repr\_tree.py | Genoma = fórmula en preorden. Grow/Full, cruce y mutación de subárbol, mutación puntual, límite de profundidad | §8 |
| repr\_gep.py | Genoma = cabeza + cola. Decodificación Karva a preorden, mutación, IS, RIS, recombinaciones | §8, D25 |
| individual.py | Individuo (máscara, K genomas, interruptores), reparación de D13 y D24, construcción de Xt′, complejidad | D13, D24 |
| fitness.py | CV con folds fijos, pipeline escalado + M por fold, MAE, caché por huella | D6, D8, D16, D18 |
| evolution.py | Bucle común: torneo con parsimonia lexicográfica, elitismo, operadores de bits, hall of fame, estadísticas por generación | §8, D17, D7 |
| transformer.py | `EvoFeatureTransformer` con `fit` / `transform` | D9, RF-03 |
| models.py | Fábrica de M (OLS) y de modelos secundarios con hiperparámetros fijos | D10, D11 |
| experiment.py | Una ejecución = dataset × configuración × representación × semilla; evaluación final y escritura del JSON | RF-09, RF-10, RF-11 |

*Por qué una representación común (formula.py):* GEP se decodifica a la misma tupla en preorden que usan los árboles. Así evaluación, impresión, tamaño y caché son idénticos para ambas representaciones y la comparación de D25 solo difiere en operadores. *Por qué preorden en una tupla:* se evalúa con una pila recorriéndola al revés, sin recursión, y un subárbol es un tramo contiguo, así que el cruce de subárbol es cortar y pegar.

### 15.2 Interfaces

```python
# formula.py
Formula = tuple[str, ...]                 # preorden, p. ej. ("mul", "x3", "log", "x1")
def evaluate(f: Formula, X: np.ndarray) -> np.ndarray          # (m,) siempre finito
def depth(f: Formula) -> int
def size(f: Formula) -> int
def to_infix(f: Formula, names: list[str]) -> str

# repr_tree.py / repr_gep.py: misma interfaz
class Representation(Protocol):
    def random_genome(self, rng: np.random.Generator) -> Genome
    def to_formula(self, g: Genome) -> Formula
    def crossover(self, a: Genome, b: Genome, rng) -> tuple[Genome, Genome]
    def mutate(self, g: Genome, rng) -> Genome

# individual.py
@dataclass(frozen=True)
class Individual:
    mask: np.ndarray            # bool (n,)  originales en Xt′
    genomes: tuple[Genome, ...] # K fórmulas
    switches: np.ndarray        # bool (K,)  fórmulas en Xt′
def repair(ind, n, rng) -> Individual          # 1 ≤ columnas ≤ n − 1 (D13, D24)
def build_matrix(ind, X, rep) -> np.ndarray     # Xt′ = originales activos + fórmulas activas
def complexity(ind, rep) -> int                 # columnas + nodos activos (D17)
def phenotype_key(ind, rep) -> str              # huella para la caché (D18)

# fitness.py
class Fitness:
    def __init__(self, X, y, folds, model_factory, rep): ...
    def __call__(self, ind) -> float            # MAE medio en CV, con caché

# transformer.py
class EvoFeatureTransformer:
    def __init__(self, cfg: RunConfig, seed: int): ...
    def fit(self, X, y) -> "EvoFeatureTransformer"   # escalado + evolución + re-selección
    def transform(self, X) -> np.ndarray              # solo aplica
    best_: Individual; formulas_: list[str]; history_: list[dict]
```

- El escalado de los originales (D2, antes de la síntesis) lo ajusta y aplica `EvoFeatureTransformer`. El escalado de Xt′ (D2, después) va dentro del pipeline de M, para que en la CV se ajuste por fold (D8).
- El intercambio de árboles completos (§8) y la recombinación de genes de GEP son la misma operación a nivel de individuo, así que viven en evolution.py, no en cada representación.

### 15.3 Configuración

Cada experimento es un archivo TOML en `configs/`. *Por qué TOML:* Python 3.11 lo lee sin dependencias (`tomllib`), admite comentarios y es legible; así cada resultado se puede rastrear hasta el archivo exacto que lo generó (RNF-01).

```toml
[experiment]
name = "final_C_tree"
datasets = ["diabetes", "california"]
configuration = "C"          # baseline | A | B | C  (§9)
representation = "tree"       # tree | gep            (D25)
seeds = { start = 0, count = 30 }

[representation]
K = 5                         # D13, lo fija el piloto
max_depth = 3                 # D13, lo fija el piloto
gep_head = 7                  # se deduce de max_depth (D20)
functions = ["add", "sub", "mul", "div", "sqrt", "log"]

[evolution]
pop_size = 100
generations = 100
tournament_size = 3
elitism = 2
p_crossover = 0.8
p_mutation = 0.2
max_columns = "n-1"           # D24

[fitness]
model = "ols"
cv_folds = 5
epsilon = 0.0                 # D17, valor fijado por el piloto

[final_selection]
hof_size = 10
repeats = 3                   # 3 × 5-fold con folds nuevos (D7)
```

### 15.4 Registro de cada ejecución

Un JSON por ejecución en `results/runs/<experimento>/<dataset>/seed_<s>.json`. *Por qué un archivo por ejecución:* si algo se corta, se relanza solo lo que falta (el script salta los archivos que ya existen), y el análisis lee todos los JSON sin depender de cómo se lanzaron.

| Campo | Contenido |
| --- | --- |
| `meta` | experimento, dataset, configuración, representación, semilla, hash del TOML, versión de Python y librerías, fecha |
| `baseline_mae` | e (M con los originales) |
| `mae` | e′ (M con Xt′) |
| `relative_improvement` | (e − e′) / e |
| `cv_mae_best` | fitness en CV del individuo final, para comparar con e′ (sobreajuste) |
| `n_columns`, `n_originals_kept`, `n_synthesized` | tamaño de Xt′ y su composición |
| `kept_originals`, `formulas` | nombres de las columnas originales que quedan y fórmulas en notación infija |
| `secondary` | e y e′ para Ridge, KNN y Random Forest (D11) |
| `history` | por generación: mejor fitness, mediana, complejidad del mejor, aciertos de caché |
| `evaluations`, `cache_hits`, `seconds` | coste de la ejecución |

### 15.5 Valores concretos

| Elemento | Valor | Por qué |
| --- | --- | --- |
| Python | ≥ 3.11 | `tomllib` incluido |
| Dependencias | numpy, scipy, scikit-learn, matplotlib, pytest; versiones fijadas en pyproject.toml | Librerías estándar; versiones fijas para reproducir (RNF-01) |
| Semilla de partición | 2026, igual para todos los experimentos | Un único e por dataset (D1) |
| Semillas del piloto | 100, 101, 102 | Distintas de las finales: los parámetros no se eligen con las mismas ejecuciones que se informan |
| Semillas finales | 0 a 29 | 30 ejecuciones (§8) |
| Folds de la evolución | `KFold(5, shuffle=True, random_state=semilla)` | Fijos dentro de una ejecución (D6) y distintos entre ejecuciones, para que la variabilidad entre semillas incluya la elección de folds |
| Folds de la re-selección | 3 repeticiones con `random_state = semilla + 10 000 + r` | Particiones nuevas, distintas de las de la evolución (D7) |
| Ridge | α = 1,0 | Valor por defecto de scikit-learn; no se ajusta para que la comparación sea de datos, no de hiperparámetros |
| KNN | k = 5, pesos uniformes | Valor por defecto, mismo motivo |
| Random Forest | 200 árboles, resto por defecto, `random_state = semilla` | Suficientes árboles para que el resultado sea estable entre ejecuciones |
| ÷ protegida | si \|denominador\| < 10⁻⁶ devuelve 1 | Convención clásica de Koza (D14) |
| Paralelismo | Una ejecución por proceso (núcleos − 1), cada una con un solo hilo de álgebra lineal | Las semillas son independientes; un hilo por proceso evita que compitan entre sí y mantiene los tiempos comparables |

### 15.6 Hitos y criterios de aceptación

Se avanza un hito cada vez. Un hito está terminado cuando pasan sus tests y se ha hecho commit; después se para para revisión antes de empezar el siguiente. *Por qué:* los errores de un hito no se arrastran a los siguientes, y cada pieza se entiende antes de construir encima.

| Hito | Qué se construye | Criterio de aceptación |
| --- | --- | --- |
| 0 | Esqueleto: pyproject, CLAUDE.md, docs/DISENO.md, carpetas | `pip install -e .` funciona y `pytest` pasa (vacío) |
| 1 | data.py y `make_splits.py` | Particiones disjuntas que cubren todas las filas; misma semilla, mismos archivos; y no aparece en X; un resumen (filas, columnas, objetivo, faltantes) que confirma o corrige la tabla del §3. Si los CSV ya traen partición o la columna objetivo no está clara, se para y se pregunta (D1) |
| 2 | primitives.py y formula.py | Las funciones protegidas dan valores finitos con 0, ±10⁻¹², ±10¹² y negativos; `evaluate` coincide con un cálculo a mano en tres fórmulas; `depth`, `size` y `to_infix` correctos |
| 3 | repr\_tree.py | Grow y Full respetan la profundidad máxima; tras 10 000 cruces y mutaciones aleatorios ningún árbol la supera y todos son preorden válido (aridades cuadran) |
| 4 | repr\_gep.py | Decodificación de los ejemplos de las diapositivas: `+/Q*c-abde` → (a·b)/c + √(d − e); `*-/Qb+b+aaab` → (√(a + b) − b)·((a + a)/b); en `/aQ/b*ab/Qa*b*-ababaababbabbbba` solo se expresan 8 símbolos. Todo cromosoma aleatorio decodifica; tras mutar, la cola sigue teniendo solo terminales |
| 5 | individual.py y fitness.py | Tras reparar 10 000 individuos aleatorios, todos tienen entre 1 y n − 1 columnas; el fitness de un individuo solo con originales coincide con un `cross_val_score` manual con los mismos folds; la caché devuelve el mismo valor |
| 6 | evolution.py | Problema sintético (5 atributos, 500 filas, y = x₁·x₂ + ruido pequeño): la configuración C encuentra una fórmula equivalente a x₁·x₂ en al menos 8 de 10 semillas; el mejor fitness nunca empeora; misma semilla, mismo resultado |
| 7 | transformer.py, models.py, experiment.py | Una ejecución completa en Diabetes con presupuesto pequeño produce un JSON con todos los campos del §15.4; `transform` sobre validación da valores finitos; un test comprueba que solo experiment.py llama a `load_validation` (RNF-02) |
| 8 | `run_pilot.py` | Tabla de la rejilla K × profundidad con MAE en CV, complejidad y tiempos; propuesta de K, profundidad y ε. **Se para:** tú fijas los valores y se escribe docs/HIPOTESIS.md con fecha antes del hito 9 |
| 9 | `run_experiments.py` | Todas las configuraciones × datasets × 30 semillas, más la variante GEP de C; relanzar el script no repite ejecuciones ya hechas |
| 10 | `analyze.py` | Genera las tablas, los tests y las figuras del §11 a partir de los JSON |

### 15.7 Comandos

```bash
pip install -e ".[dev]"
pytest
python scripts/make_splits.py                      # hito 1, una sola vez
python scripts/run_pilot.py --dataset california   # hito 8
python scripts/run_experiments.py configs/final_C_tree.toml
python scripts/analyze.py results/runs             # hito 10
```
