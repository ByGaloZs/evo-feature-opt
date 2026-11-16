# Hipótesis (D21, D25)

Fecha de redacción: **2026-11-16**. Este archivo se escribe y se commitea antes de lanzar el
hito 9 (experimentos finales). No se edita después de ver resultados finales.

H1–H4 son las del diseño (`docs/DISENO.md`, D21 y D25) y se formularon antes de ejecutar nada.
H5 se formula **después del piloto y solo con train**, y así se marca.

## Hipótesis del diseño (antes de ejecutar)

- **H1.** En California, C mejora claramente (e′ < e), porque sus relaciones son no lineales.
- **H2.** En Diabetes, la mejora es pequeña o nula, porque la relación es casi lineal, hay pocas
  filas y la validación es ruidosa.
- **H3.** B aporta más que A en ambos datasets (crear atributos aporta más que quitarlos).
- **H4.** No habrá diferencia significativa en e′ entre árboles y GEP (configuración C, mismo
  tamaño máximo de fórmula), porque con el mismo tamaño máximo ambas pueden expresar
  prácticamente las mismas fórmulas; si hay diferencias, aparecerán en la velocidad de
  convergencia.

## Hipótesis formulada tras el piloto (solo con train)

- **H5 (formulada tras el piloto, solo con train).** En California la restricción de D24 está
  activa: en 26 de las 27 ejecuciones del piloto (y en las 3 con K = 3 y profundidad 3) los
  individuos finales usan n − 1 = 7 columnas; la restante usa 6. Se espera que B (todos los
  originales más síntesis, sin límite de columnas) obtenga un e′ menor que C, y que la
  diferencia mida el coste de exigir s < n.

## Contexto del piloto disponible antes de ejecutar (solo train, MAE en CV repetida con folds nuevos)

Se anota para que quede constancia de lo que se sabía al fijar las hipótesis; no modifica H1–H4.

| Dataset | Baseline (OLS, todos los originales) | C con K = 3, profundidad 3 (ε = 0) | Mejora relativa en CV |
| --- | --- | --- | --- |
| California | 0,5362 | 0,4823 | 10,0 % |
| Diabetes | 46,07 | 43,74 | 5,1 % |

- Con el ε final (Diabetes 1,26, California 0,0030) el MAE en CV pasa a 0,4852 (+0,0029; la mejora
  baja a 9,5 %) en California y a 44,43 (+0,69; la mejora baja a 3,6 %) en Diabetes. En ninguno
  de los dos empeora más de ε, así que se sigue adelante con los ε finales.
- La mejora en CV de Diabetes (≈ 5 %) es mayor que la que sugiere H2. H2 se mantiene tal como se
  escribió: la CV interna no es la validación, y la validación de Diabetes tiene solo 88 filas.
- Parámetros fijados por el piloto (D20): K = 3 y profundidad máxima 3 para ambos datasets
  (GEP: h = 7); ε por dataset: Diabetes 1,26 y California 0,0030.
