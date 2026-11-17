"""Reúne todos los resultados en un solo documento de consulta: ``docs/RESULTADOS.md``.

No calcula nada nuevo: ensambla, en orden cronológico, lo que ya generaron los demás scripts y la
interpretación manual, para no transcribir cifras a mano:

1. Resumen: contraste de H1–H5 (de ``results/analysis/estadistica.json``) y tabla e / e′.
2. Piloto (hito 8, solo train): ``results/pilot/<dataset>/summary.json`` y ``eps_check.json``.
3. Hipótesis: ``docs/HIPOTESIS.md``, escritas antes de lanzar los experimentos finales.
4. Experimentos finales (hito 9–10): ``results/analysis/informe.md`` y sus figuras.
5. Análisis posteriores a la validación: ``results/d23/resumen.md`` (D23), el diagnóstico de la cola de
   C en California (``results/diagnostics``) y ``docs/INTERPRETACION.md`` (manual).

El archivo resultante se genera; no se edita a mano. Para actualizarlo tras cambiar alguna fuente:
``python scripts/build_resultados.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "RESULTADOS.md"

ANALISIS = RAIZ / "results" / "analysis"
PILOTO = RAIZ / "results" / "pilot"
D23 = RAIZ / "results" / "d23" / "resumen.md"
DIAGNOSTICO = RAIZ / "results" / "diagnostics" / "california_c_tree_tail.md"
INTERPRETACION = RAIZ / "docs" / "INTERPRETACION.md"

ETIQUETA = {"baseline": "Línea base", "A": "A (selección)", "B": "B (síntesis)", "C_tree": "C (árbol)", "C_gep": "C (GEP)"}
NOMBRE_DATASET = {"diabetes": "Diabetes", "california": "California"}
FIGURAS = (("cajas_e_prime.png", "Diagramas de caja de e′ con la línea de e"),
           ("convergencia.png", "Convergencia: mejor MAE en CV por generación (mediana y RIC)"),
           ("convergencia_arbol_vs_gep.png", "Convergencia: C con árboles frente a C con GEP"),
           ("originales_conservados.png", "Fracción de ejecuciones que conserva cada original"))


def _cargar_pilot_module():
    spec = importlib.util.spec_from_file_location("run_pilot_aux", RAIZ / "scripts" / "run_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def leer(ruta: Path) -> str:
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta.relative_to(RAIZ)}: genera antes sus resultados")
    return ruta.read_text()


def sin_titulo(texto: str) -> str:
    """Quita el título de nivel 1 de un documento (el ensamblado pone el suyo)."""
    lineas = texto.splitlines()
    while lineas and not lineas[0].startswith("# "):
        if lineas[0].strip():
            break
        lineas = lineas[1:]
    if lineas and lineas[0].startswith("# "):
        lineas = lineas[1:]
    return "\n".join(lineas).strip("\n")


def bajar_niveles(texto: str, n: int = 1) -> str:
    """Baja n niveles los títulos Markdown (## → ###) sin tocar las líneas dentro de bloques de código."""
    salida, en_codigo = [], False
    for linea in texto.splitlines():
        if linea.startswith("```"):
            en_codigo = not en_codigo
        if not en_codigo and linea.startswith("#"):
            linea = "#" * n + linea
        salida.append(linea)
    return "\n".join(salida)


def resumen(estadistica: dict) -> str:
    partes = ["Contraste de las hipótesis de `docs/HIPOTESIS.md` con los experimentos finales (30 semillas por celda, "
              "evaluados en validación):", ""]
    for h, x in estadistica["verdicts"].items():
        estado = "interpretación manual" if x["supported"] is None else ("apoyada" if x["supported"] else "no apoyada")
        partes.append(f"- **{h}** — {x['text']} → **{estado}**. {x['evidence']}." + (f" ({x['note']})" if "note" in x else ""))
    partes += ["", "**Conclusión de H2 tras D23 (posterior a la validación, manual):** ver la sección 5.3. "
                   "La mejora de Diabetes en la partición 2026 no se reproduce en otras particiones.", "",
               "Resumen de e frente a e′ (mediana de e′ sobre las 30 semillas):", "",
               "| Dataset | Configuración | e | mediana e′ | % e′ < e | mejora rel. mediana |", "| --- | --- | --- | --- | --- | --- |"]
    for ds, porclave in estadistica["descriptive"].items():
        for k, d in porclave.items():
            partes.append(f"| {NOMBRE_DATASET.get(ds, ds)} | {ETIQUETA[k]} | {d['e']:.4g} | {d['e_prime_median']:.4g} | "
                          f"{d['success_pct']:.0f} | {100 * d['rel_improvement_median']:.2f} % |")
    return "\n".join(partes)


def piloto() -> str:
    pil = _cargar_pilot_module()
    partes = []
    for ds in ("diabetes", "california"):
        resumen_ds = json.loads(leer(PILOTO / ds / "summary.json"))
        partes += [bajar_niveles(f"## {NOMBRE_DATASET[ds]}", 1), "", pil.tabla(resumen_ds, ds), ""]
    partes += ["### Comprobación de ε > 0 (hito 8b)", ""]
    for ds in ("diabetes", "california"):
        partes += [pil.tabla_epsilon(json.loads(leer(PILOTO / ds / "eps_check.json"))), ""]
    partes += ["**Decisión tras el piloto** (ver `docs/DISENO.md`, D20): K = 3 y profundidad 3 en ambos datasets (GEP: h = 7); "
               "ε = 1,26 en Diabetes y 0,0030 en California."]
    return "\n".join(partes)


def finales() -> str:
    informe = leer(ANALISIS / "informe.md")
    marca = "## e frente a e′ por dataset y configuración"
    if marca not in informe:
        raise ValueError("informe.md no tiene la sección esperada; regenera con scripts/analyze.py")
    cuerpo = informe[informe.index(marca):]
    if "\n## Figuras" in cuerpo:
        cuerpo = cuerpo[: cuerpo.index("\n## Figuras")]
    partes = [bajar_niveles(cuerpo.strip("\n"), 1), "", "### Figuras", ""]
    for archivo, descripcion in FIGURAS:
        ruta = ANALISIS / "figuras" / archivo
        if ruta.exists():
            partes += [f"**{descripcion}**", "", f"![{descripcion}](../results/analysis/figuras/{archivo})", ""]
    return "\n".join(partes)


def construir() -> str:
    estadistica = json.loads(leer(ANALISIS / "estadistica.json"))
    d23 = bajar_niveles(sin_titulo(leer(D23)), 1)
    diag = bajar_niveles(sin_titulo(leer(DIAGNOSTICO)), 1)
    interp = bajar_niveles(sin_titulo(leer(INTERPRETACION)), 1)
    partes = [
        "# Resultados",
        "",
        "Documento de consulta que reúne todos los resultados del proyecto. **Se genera con `scripts/build_resultados.py`; no se edita a mano.** "
        "Cada sección procede de un archivo que ya existía; si cambia alguno, se vuelve a ejecutar el script.",
        "",
        "| Sección | Qué contiene | Fuente | Momento |",
        "| --- | --- | --- | --- |",
        "| 1. Resumen | Contraste de H1–H5 y tabla e / e′ | `results/analysis/estadistica.json` | validación |",
        "| 2. Piloto | Rejilla K × profundidad, ε y comprobación de ε > 0 | `results/pilot/` | solo train |",
        "| 3. Hipótesis | H1–H5 tal como se escribieron antes de lanzar los experimentos | `docs/HIPOTESIS.md` | antes de validación |",
        "| 4. Experimentos finales | Tablas, tests estadísticos, modelos secundarios, fórmulas y figuras | `results/analysis/` | validación |",
        "| 5. Posterior a la validación | D23, diagnóstico de la cola de California e interpretación de H2 | `results/d23/`, `results/diagnostics/`, `docs/INTERPRETACION.md` | después de ver la validación |",
        "",
        "Los análisis de la sección 5 se hicieron después de ver los resultados de validación, sin cambiar el código del sistema ni "
        "ningún parámetro; se informan salgan como salgan (`docs/DISENO.md`, D23).",
        "",
        "## 1. Resumen", "",
        resumen(estadistica), "",
        "## 2. Piloto (hito 8, solo train)", "",
        piloto(), "",
        "## 3. Hipótesis (escritas antes de lanzar los experimentos)", "",
        bajar_niveles(sin_titulo(leer(RAIZ / "docs" / "HIPOTESIS.md")), 1), "",
        "## 4. Experimentos finales (validación)", "",
        finales(), "",
        "## 5. Análisis posteriores a la validación", "",
        "### 5.1 D23: robustez a la partición en Diabetes", "",
        bajar_niveles(d23, 1), "",
        "### 5.2 Diagnóstico de la cola de C con árboles en California", "",
        bajar_niveles(diag, 1), "",
        "### 5.3 Interpretación de H2 (manual)", "",
        bajar_niveles(interp, 1), "",
    ]
    return "\n".join(partes)


def main() -> None:
    texto = construir()
    SALIDA.write_text(texto)
    print(f"Escrito {SALIDA.relative_to(RAIZ)} ({len(texto.splitlines())} líneas)")


if __name__ == "__main__":
    main()
