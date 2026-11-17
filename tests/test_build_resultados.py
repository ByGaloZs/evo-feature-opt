"""Tests de docs/RESULTADOS.md: documento único que reúne todos los resultados (generado)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_resultados", RAIZ / "scripts" / "build_resultados.py")
br = importlib.util.module_from_spec(_spec)
sys.modules["build_resultados"] = br
_spec.loader.exec_module(br)


def test_el_documento_commiteado_esta_al_dia_con_sus_fuentes():
    """Si cambia alguna fuente sin regenerar, este test falla: ejecuta scripts/build_resultados.py."""
    assert br.SALIDA.read_text() == br.construir()


def test_secciones_en_orden_cronologico():
    texto = br.construir()
    titulos = re.findall(r"^## (\d\. .+)$", texto, re.M)
    assert [t.split(".")[0] for t in titulos] == ["1", "2", "3", "4", "5"]
    assert titulos[1].startswith("2. Piloto") and titulos[3].startswith("4. Experimentos finales")
    assert "### 5.1 D23" in texto and "### 5.2 Diagnóstico" in texto and "### 5.3 Interpretación de H2" in texto
    assert texto.count("\n# ") == 0 and texto.startswith("# Resultados")  # un solo título de nivel 1


def test_incluye_las_cifras_de_cada_fuente():
    texto = br.construir()
    est = json.loads((RAIZ / "results" / "analysis" / "estadistica.json").read_text())
    c = est["descriptive"]["california"]["C_tree"]
    assert f"{c['e_prime_median']:.4g}" in texto and f"{c['e']:.4g}" in texto
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert f"**{h}**" in texto
    d23 = json.loads((RAIZ / "results" / "d23" / "resumen.json").read_text())
    assert f"{100 * d23['across_new_partitions']['rel_improvement_median']:.2f} %" in texto
    assert "0,0053" in texto  # interpretación manual de H2
    assert "california_c_tree_tail" not in texto  # el diagnóstico se integra, no solo se enlaza
    assert "Hall of fame de las ejecuciones malas" in texto
    assert "Comprobación 8b" in texto and "K = 3" in texto


def test_las_figuras_enlazadas_existen():
    texto = br.construir()
    enlaces = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", texto)
    assert len(enlaces) == 4
    for e in enlaces:
        assert (RAIZ / "docs" / e).resolve().exists(), e


def test_las_fuentes_no_se_modifican_al_construir():
    fuentes = [br.ANALISIS / "informe.md", br.D23, br.DIAGNOSTICO, br.INTERPRETACION, RAIZ / "docs" / "HIPOTESIS.md"]
    antes = [f.read_bytes() for f in fuentes]
    br.construir()
    assert [f.read_bytes() for f in fuentes] == antes


def test_bajar_niveles_respeta_bloques_de_codigo():
    assert br.bajar_niveles("## a\n```\n# no\n```\n### b", 1) == "### a\n```\n# no\n```\n#### b"
    assert br.sin_titulo("# Titulo\n\ntexto\n## sec") == "texto\n## sec"


def test_seccion_5_4_solo_aparece_si_el_exploratorio_esta_ejecutado(tmp_path, monkeypatch):
    assert ("### 5.4 Exploratorio" in br.construir()) == br.EXPLORA_RESUMEN.exists()  # solo si ya se ha ejecutado
    resumen = tmp_path / "resumen.md"
    resumen.write_text("Exploratorio.\n\n| Brazo | n |\n| --- | --- |\n| pm0.5 | 30 |\n")
    doc = tmp_path / "EXPLORATORIO.md"
    doc.write_text("# T\n\n## H6 (exploratoria)\n\nTexto de H6.\n\n## Diseño\n\nx\n\n## Resultados e interpretación\n\nInterpretación manual.\n")
    monkeypatch.setattr(br, "EXPLORA_RESUMEN", resumen)
    monkeypatch.setattr(br, "EXPLORATORIO", doc)
    texto = br.construir()
    assert "### 5.4 Exploratorio" in texto and "Texto de H6." in texto and "| pm0.5 | 30 |" in texto and "Interpretación manual." in texto
    assert "| 5.4 Exploratorio |" in texto
    assert texto.index("### 5.3") < texto.index("### 5.4")
    assert br.seccion("## a\nuno\n\n## b\ndos", "a") == "uno"


def test_nota_de_h4_en_la_seccion_1():
    texto = br.construir()
    resumen = texto[texto.index("## 1. Resumen"): texto.index("## 2. Piloto")]
    nota = ("**H4 tras los análisis posteriores (5.2 y 5.4):** misma mediana de e′, pero los árboles fallan más "
            "(11/30 ejecuciones malas en California, fallo de búsqueda); con más mutación la cola casi desaparece "
            "(descriptivo, no significativo). La diferencia de fiabilidad se atribuye probablemente a la intensidad de "
            "mutación, no a la representación.")
    assert nota in resumen
    # tras la nota de H2 y antes de la tabla de e / e′
    assert resumen.index("**Conclusión de H2 tras D23") < resumen.index(nota) < resumen.index("Resumen de e frente a e′")


def test_5_4_sin_tabla_repetida_ni_remision_a_si_misma():
    texto = br.construir()
    sec = texto[texto.index("### 5.4 Exploratorio"):]
    assert "Lo observado (solo train)" not in sec
    assert "en la sección 5.4 de" not in sec and "Tabla y tests en" not in sec
    interp = sec[sec.index("#### Interpretación"):]
    assert "| `p_mutation` 0,2 (control) |" not in interp  # la tabla reducida ya no está repetida
    assert interp.count("|") == 0  # sin tablas en la interpretación
    # el resto de la interpretación se conserva
    assert "El patrón descriptivo es el que predecía H6" in interp and "**Qué no se puede afirmar.**" in interp
    assert "Esto no cambia los resultados finales" in interp and "Ejecutado el 2026-11-17: 60 ejecuciones, 0 fallos." in interp
    # la tabla completa y los tests siguen en la parte de resultados de la 5.4
    assert "| pm0.5 | control (0,2) |" in sec and "| pm0.2 (control) |" in sec


def test_exploratorio_md_conserva_su_texto_original():
    """Solo se limpia el texto en RESULTADOS.md; docs/EXPLORATORIO.md no se toca."""
    doc = br.EXPLORATORIO.read_text()
    assert "**Lo observado (solo train).**" in doc and "sección 5.4 de\n`docs/RESULTADOS.md`" in doc


def test_limpiar_interpretacion_es_tolerante():
    assert br.limpiar_interpretacion("Sin nada que quitar.") == "Sin nada que quitar."
