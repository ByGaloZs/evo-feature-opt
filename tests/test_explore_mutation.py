"""Tests del análisis exploratorio de la tasa de mutación (solo train, posterior a la validación)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from evofeat import data
from evofeat.transformer import load_config

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_explore_mutation", RAIZ / "scripts" / "run_explore_mutation.py")
em = importlib.util.module_from_spec(_spec)
sys.modules["run_explore_mutation"] = em  # los procesos hijos deben poder importar las funciones
_spec.loader.exec_module(em)

PEQUENO = dict(pop_size=12, generations=2, hof_size=3, repeats=1)


def _instantanea(*dirs: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


# --- el script solo usa train --------------------------------------------------------------------

def test_el_script_no_llama_a_load_validation_ni_a_experiment():
    texto = (RAIZ / "scripts" / "run_explore_mutation.py").read_text()
    assert "load_validation" not in texto
    assert "from evofeat.experiment" not in texto and "import experiment" not in texto
    assert "load_train" in texto


def test_el_ajuste_no_lee_validacion(tmp_path, monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("no se puede leer validación")

    monkeypatch.setattr(data, "load_validation", prohibido)
    ruta, hecha = em.ejecutar_semilla(0.5, 0, tmp_path, PEQUENO)
    reg = json.loads(Path(ruta).read_text())
    assert hecha and reg["cv_mae_best"] > 0
    assert not any("mae" == k or k == "e_prime" or k == "baseline_mae" for k in reg)  # no se calcula e′
    assert reg["meta"]["p_mutation"] == 0.5 and reg["meta"]["dataset"] == "california"


# --- configuración: todo igual salvo p_mutation ------------------------------------------------------

def test_brazos_son_la_config_final_con_otro_p_mutation():
    final = load_config(str(RAIZ / "configs" / "final_C_tree.toml"))
    assert (em.BRAZOS, em.P_CONTROL, em.SEMILLAS) == ((0.5, 1.0), 0.2, tuple(range(30)))
    for p in (0.2, 0.5, 1.0):
        cfg, h = em.config_brazo(p)
        assert cfg.p_mutation == p and cfg.epsilon == 0.003 and (cfg.K, cfg.max_depth) == (3, 3)
        assert replace(cfg, name=final.name, p_mutation=final.p_mutation, epsilon=final.epsilon) == final
        assert cfg.configuration == "C" and cfg.representation == "tree"
    assert em.config_brazo(0.5)[1] != em.config_brazo(1.0)[1]
    assert load_config(str(RAIZ / "configs" / "final_C_tree.toml")).p_mutation == 0.2  # la config final no cambia


def test_plan_y_estimacion():
    assert len(em.pendientes(Path("/ruta/inexistente/explore"))) == 60  # 2 brazos × 30 semillas
    est = em.estimar(jobs=5, base=Path("/ruta/inexistente/explore"))
    assert (est["runs"], est["arm_runs"], est["control_check_runs"]) == (63, 60, 3)
    assert est["wall_minutes_low"] < est["wall_minutes_high"]


# --- comprobación del control y ejecución con presupuesto pequeño ---------------------------------------

@pytest.fixture(scope="module")
def mundo(tmp_path_factory):
    """Un 'final' de mentira (misma maquinaria, presupuesto mínimo) y una base de exploración."""
    raiz = tmp_path_factory.mktemp("explore")
    finales = raiz / "finales"
    for s in (0, 1):
        em.ejecutar_semilla(em.P_CONTROL, s, finales, PEQUENO)
    return raiz, finales / "runs" / "pm0.2" / "california"


def test_control_coincide_exactamente_si_la_simulacion_es_determinista(mundo, tmp_path):
    raiz, dir_final = mundo
    filas = em.comprobar_control(tmp_path / "base", dir_final, (0, 1), PEQUENO)
    assert all(f["equal"] for f in filas) and len(filas) == 2
    assert filas[0]["final_cv_mae_best"] == filas[0]["rerun_cv_mae_best"]  # igualdad exacta, no aproximada


def test_si_el_control_no_coincide_se_para_y_no_se_lanzan_los_brazos(mundo, tmp_path):
    import shutil
    raiz, dir_final = mundo
    copia = tmp_path / "final_manipulado"
    shutil.copytree(dir_final, copia)
    f = copia / "seed_0.json"
    r = json.loads(f.read_text())
    r["cv_mae_best"] += 1e-12  # una diferencia mínima basta
    f.write_text(json.dumps(r))
    base = tmp_path / "base"
    with pytest.raises(em.ControlNoCoincide, match="semilla 0"):
        em.lanzar(base, jobs=1, brazos=(0.5,), semillas=(0,), semillas_comprobacion=(0,), overrides=PEQUENO, dir_final=copia, verbose=False)
    assert not (base / "runs").exists()  # los brazos no se ejecutaron
    assert json.loads((base / "control_check" / "comprobacion.json").read_text())[0]["equal"] is False


def test_lanzar_es_reanudable_y_solo_escribe_bajo_la_base(mundo, tmp_path):
    raiz, dir_final = mundo
    antes = _instantanea(RAIZ / "data", RAIZ / "results" / "runs", RAIZ / "configs", RAIZ / "results" / "analysis", RAIZ / "results" / "d23")
    base = tmp_path / "base"
    kw = dict(jobs=1, brazos=(0.5, 1.0), semillas=(0, 1), semillas_comprobacion=(0, 1), overrides=PEQUENO, dir_final=dir_final, verbose=False)
    r = em.lanzar(base, **kw)
    assert (r["pendientes"], r["hechas"], r["fallos"]) == (4, 4, [])
    assert {p.relative_to(base).parts[0] for p in base.rglob("*") if p.is_file()} == {"runs", "control_check"}
    assert (base / "runs" / "pm1" / "california" / "seed_1.json").exists()
    assert _instantanea(RAIZ / "data", RAIZ / "results" / "runs", RAIZ / "configs", RAIZ / "results" / "analysis", RAIZ / "results" / "d23") == antes
    contenido = {p: p.read_bytes() for p in base.rglob("seed_*.json")}
    r2 = em.lanzar(base, **kw)
    assert (r2["pendientes"], r2["hechas"]) == (0, 0)
    assert {p: p.read_bytes() for p in base.rglob("seed_*.json")} == contenido  # no se reescribe nada


def test_un_hash_distinto_detiene_el_plan(mundo, tmp_path):
    raiz, dir_final = mundo
    base = tmp_path / "base"
    em.ejecutar_semilla(0.5, 0, base, PEQUENO)
    ruta = em.ruta_json(base, 0.5, 0)
    r = json.loads(ruta.read_text())
    r["meta"]["config_hash"] = "otro"
    ruta.write_text(json.dumps(r))
    with pytest.raises(RuntimeError, match="otra configuración"):
        em.pendientes(base, (0.5,), (0,), PEQUENO)


# --- resumen y tests exploratorios ---------------------------------------------------------------------------

def _reg(seed, cv, formulas, cplx=10, hist=None, final=False):
    r = {"meta": {"seed": seed}, "cv_mae_best": cv, "formulas": formulas, "distinct_evaluations": 5000, "seconds_fit": 100.0,
         "history": hist or [{"best_fitness": x} for x in (0.54, 0.51, 0.49)]}
    if final:
        r["hall_of_fame"] = [{"complexity": cplx}]
    else:
        r["complexity"] = cplx
    return r


def _escribir(carpeta: Path, regs):
    carpeta.mkdir(parents=True, exist_ok=True)
    for r in regs:
        (carpeta / f"seed_{r['meta']['seed']}.json").write_text(json.dumps(r))


def test_resumen_y_tests_contra_scipy(tmp_path):
    malas, buenas = ["(AveOccup + MedInc)"], ["sqrt(|AveOccup|)", "log(1 + |AveRooms|)"]
    control = [_reg(s, 0.52 if s < 12 else 0.48, malas if s < 12 else buenas, final=True) for s in range(30)]  # 12/30 malas
    arm = [_reg(s, 0.52 if s < 4 else 0.48, malas if s < 4 else buenas) for s in range(30)]  # 4/30 malas
    gep = [_reg(s, 0.52 if s < 2 else 0.48, malas if s < 2 else buenas, final=True) for s in range(30)]  # 2/30 malas
    _escribir(tmp_path / "control", control)
    _escribir(tmp_path / "base" / "runs" / "pm0.5" / "california", arm)
    _escribir(tmp_path / "gep", gep)
    r = em.resumir(tmp_path / "base", tmp_path / "control", tmp_path / "gep", brazos=(0.5,))
    assert list(r["arms"]) == ["pm0.2 (control)", "pm0.5", "GEP (referencia)"]
    a = r["arms"]["pm0.5"]
    assert a["n"] == 30 and a["n_cv_below_threshold"] == 26 and a["pct_cv_below_threshold"] == pytest.approx(100 * 26 / 30)
    assert a["pct_transforms_AveOccup"] == pytest.approx(100 * 26 / 30) and a["pct_log_AveRooms"] == pytest.approx(100 * 26 / 30)
    assert r["arms"]["pm0.2 (control)"]["pct_cv_below_threshold"] == pytest.approx(100 * 18 / 30)
    assert a["complexity_median"] == 10 and r["arms"]["pm0.2 (control)"]["complexity_median"] == 10  # finales: hall_of_fame[0]
    assert a["generation_below_threshold_median"] == 2 and a["runs_reaching_threshold"] == 30  # best_fitness 0.49 < 0,5 en gen. 2
    cv = np.array([r_["cv_mae_best"] for r_ in arm])
    assert a["cv_median"] == pytest.approx(np.median(cv)) and a["cv_iqr"] == pytest.approx(np.subtract(*np.percentile(cv, [75, 25])))
    t = r["tests_exploratory_uncorrected"]["pm0.5"]
    fisher_ctrl = stats.fisher_exact([[26, 4], [18, 12]], alternative="two-sided").pvalue
    fisher_gep = stats.fisher_exact([[26, 4], [28, 2]], alternative="two-sided").pvalue
    assert t["vs_control"]["fisher_p"] == pytest.approx(fisher_ctrl) and t["vs_gep"]["fisher_p"] == pytest.approx(fisher_gep)
    mw = stats.mannwhitneyu(cv, [x["cv_mae_best"] for x in control], alternative="two-sided")
    assert t["vs_control"]["mann_whitney_p"] == pytest.approx(mw.pvalue)
    texto = em.informe(r)
    assert "sin corregir" in texto and "Fisher" in texto and "Mann-Whitney" in texto and "GEP (referencia)" in texto
    json.dumps(r, allow_nan=False)


def test_escribir_resumen_solo_bajo_la_base(tmp_path):
    _escribir(tmp_path / "control", [_reg(s, 0.5 + 0.001 * s, ["a"], final=True) for s in range(5)])
    _escribir(tmp_path / "gep", [_reg(s, 0.48, ["a"], final=True) for s in range(5)])
    _escribir(tmp_path / "base" / "runs" / "pm0.5" / "california", [_reg(s, 0.49, ["a"]) for s in range(5)])
    em.escribir_resumen(tmp_path / "base", dir_control=tmp_path / "control", dir_gep=tmp_path / "gep", brazos=(0.5,))
    assert {p.name for p in (tmp_path / "base").iterdir()} == {"runs", "resumen.json", "resumen.md"}


@pytest.mark.skipif(not (RAIZ / "results" / "runs" / "final_C_tree" / "california").exists(), reason="no hay resultados finales")
def test_las_referencias_reales_cuadran_con_la_seccion_5_2():
    arbol = em.resumen_brazo(em._cargar(em.FINAL_ARBOL))
    gep = em.resumen_brazo(em._cargar(em.FINAL_GEP))
    assert (arbol["n_cv_below_threshold"], gep["n_cv_below_threshold"]) == (21, 28)  # las cifras de la sección 5.2
    assert arbol["generation_below_threshold_median"] == 39 and gep["generation_below_threshold_median"] == 29
