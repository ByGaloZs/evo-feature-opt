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
