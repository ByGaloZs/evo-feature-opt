"""Tests de D23 (robustez a la partición) y del diagnóstico de la cola de C en California."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from evofeat import data
from evofeat.transformer import load_config

RAIZ = Path(__file__).resolve().parents[1]


def _cargar_script(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod  # los procesos hijos deben poder importar las funciones
    spec.loader.exec_module(mod)
    return mod


d23 = _cargar_script("run_d23")
dg = _cargar_script("diagnose_tail")

PEQUENO = dict(pop_size=12, generations=2, hof_size=3, repeats=1)


def _instantanea(*dirs: Path) -> dict[str, str]:
    """Hash de todos los archivos bajo los directorios dados (para comprobar que no se escribe fuera)."""
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


# =============================================================================================
# D23
# =============================================================================================

def test_semillas_de_particion_distintas_de_2026_y_entre_si():
    assert d23.SEMILLAS_PARTICION == (2027, 2028, 2029, 2030, 2031)
    assert data.SEMILLA_PARTICION == 2026 and 2026 not in d23.SEMILLAS_PARTICION
    assert len(set(d23.SEMILLAS_PARTICION)) == 5
    assert d23.SEMILLAS_EJECUCION == tuple(range(10))


def test_particiones_nuevas_son_distintas_disjuntas_y_cubren_todo(tmp_path):
    assert d23.crear_particiones(tmp_path) == [True] * 5
    original = (RAIZ / "data" / "splits" / "diabetes_val.csv").read_bytes()
    raw = data.load_raw(d23.DATASET)
    filas_raw = np.column_stack([raw.X, raw.y])
    ordenar = lambda M: M[np.lexsort(M.T[::-1])]  # noqa: E731
    vistas = []
    for s in d23.SEMILLAS_PARTICION:
        d = d23.dir_particion(tmp_path, s)
        val_bytes = (d / "diabetes_val.csv").read_bytes()
        assert val_bytes != original  # distinta de la 2026
        vistas.append(val_bytes)
        _, tr = data._leer_csv(d / "diabetes_train.csv")[0], data._leer_csv(d / "diabetes_train.csv")[1]
        va = data._leer_csv(d / "diabetes_val.csv")[1]
        assert tr.shape[0] + va.shape[0] == raw.X.shape[0]
        np.testing.assert_array_equal(ordenar(np.vstack([tr, va])), ordenar(filas_raw))  # disjuntas y completas
        assert va.shape[0] == round(0.2 * raw.X.shape[0])
    assert len(set(vistas)) == 5  # distintas entre sí


def test_crear_particiones_es_determinista_y_no_regenera(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    d23.crear_particiones(a)
    d23.crear_particiones(b)
    assert _instantanea(a).values() and list(_instantanea(a).values()) == list(_instantanea(b).values())
    assert d23.crear_particiones(a) == [False] * 5


def test_plan_son_100_ejecuciones_con_los_parametros_finales(tmp_path):
    tareas = d23.plan(tmp_path)
    assert len(tareas) == 100  # 5 particiones × 2 configuraciones × 10 semillas
    assert tareas[0][0] == "C_tree" and tareas[-1][0] == "baseline"
    for clave, ps, seed, cfg, h in tareas:
        final = load_config(str(d23.CONFIGS[clave]))
        assert replace(cfg, name=final.name, datasets=final.datasets, seed_start=final.seed_start,
                       seed_count=final.seed_count) == final  # solo cambian nombre, dataset y semillas
        assert cfg.name == f"d23_p{ps}_{clave}" and cfg.datasets == ("diabetes",) and cfg.seed_count == 10
    c = next(t[3] for t in tareas if t[0] == "C_tree")
    assert (c.K, c.max_depth, c.epsilon_for("diabetes"), c.representation, c.configuration) == (3, 3, 1.26, "tree", "C")
    assert len({t[4] for t in tareas}) == 10  # un hash por (configuración, partición)
    est = d23.estimar(tareas, jobs=5)
    assert est["runs"] == 100 and est["wall_seconds"] == pytest.approx(est["cpu_seconds"] / 5)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Dos particiones × dos semillas con presupuesto mínimo, en un directorio temporal."""
    base = tmp_path_factory.mktemp("d23")
    antes = _instantanea(RAIZ / "data", RAIZ / "results" / "runs", RAIZ / "configs")
    splits_antes = data.DIR_SPLITS
    r = d23.lanzar(base, jobs=1, particiones=(2027, 2028), seeds=(0, 1), overrides=PEQUENO, verbose=False)
    return base, r, antes, splits_antes


def test_lanzar_ejecuta_y_solo_escribe_bajo_la_base(tiny):
    base, r, antes, splits_antes = tiny
    assert (r["pendientes"], r["hechas"], r["fallos"]) == (8, 8, [])
    despues = _instantanea(RAIZ / "data", RAIZ / "results" / "runs", RAIZ / "configs")
    assert despues == antes  # ni data/splits, ni results/runs, ni configs se tocaron
    assert data.DIR_SPLITS == splits_antes  # la ruta de particiones se restaura
    escritos = {p.relative_to(base).parts[0] for p in base.rglob("*") if p.is_file()}
    assert escritos == {"splits", "runs"}
    assert (base / "runs" / "d23_p2027_C_tree" / "diabetes" / "seed_0.json").exists()


def test_cada_particion_usa_su_propio_train_y_validacion(tiny):
    base, *_ = tiny
    for ps in (2027, 2028):
        d = d23.dir_particion(base, ps)
        _, tr = data._leer_csv(d / "diabetes_train.csv")
        _, va = data._leer_csv(d / "diabetes_val.csv")
        m = LinearRegression().fit(tr[:, :-1], tr[:, -1])
        e = float(np.mean(np.abs(va[:, -1] - m.predict(va[:, :-1]))))
        reg = json.loads((base / "runs" / f"d23_p{ps}_baseline" / "diabetes" / "seed_0.json").read_text())
        assert reg["baseline_mae"] == pytest.approx(e, rel=1e-9)
        assert reg["meta"]["dataset"] == "diabetes"
    assert json.loads((base / "runs" / "d23_p2027_C_tree" / "diabetes" / "seed_0.json").read_text())["meta"]["epsilon"] == 1.26
    # las dos particiones dan un e distinto: se leyó cada una
    e1 = json.loads((base / "runs" / "d23_p2027_baseline" / "diabetes" / "seed_0.json").read_text())["baseline_mae"]
    e2 = json.loads((base / "runs" / "d23_p2028_baseline" / "diabetes" / "seed_0.json").read_text())["baseline_mae"]
    assert e1 != e2


def test_relanzar_no_repite_y_un_hash_distinto_detiene(tiny, tmp_path):
    base, *_ = tiny
    r = d23.lanzar(base, jobs=1, particiones=(2027, 2028), seeds=(0, 1), overrides=PEQUENO, verbose=False)
    assert (r["pendientes"], r["hechas"]) == (0, 0)
    copia = tmp_path / "copia"
    import shutil
    shutil.copytree(base, copia)
    ruta = copia / "runs" / "d23_p2027_C_tree" / "diabetes" / "seed_0.json"
    reg = json.loads(ruta.read_text())
    reg["meta"]["config_hash"] = "otro"
    ruta.write_text(json.dumps(reg))
    with pytest.raises(RuntimeError, match="otra configuración"):
        d23.plan(copia, (2027, 2028), (0, 1), PEQUENO)


def test_resumen_con_particiones_reales_de_prueba(tiny, tmp_path):
    base, *_ = tiny
    ref = tmp_path / "ref"
    for exp, mae, rel in (("final_baseline", 40.0, 0.0), ("final_C_tree", 38.0, 0.05)):
        for s in range(3):
            f = ref / exp / "diabetes" / f"seed_{s}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"meta": {"seed": s}, "baseline_mae": 40.0, "mae": mae, "relative_improvement": rel,
                                     "cv_mae_best": 46.0, "formulas": ["(sex * age)", "log(1 + |s6|)"] if exp == "final_C_tree" else []}))
    r = d23.resumir(base, ref, particiones=(2027, 2028), seeds=(0, 1))
    assert set(r["rows"]) == {"2026 (referencia)", "2027", "2028"}
    ref_fila = r["rows"]["2026 (referencia)"]
    assert ref_fila["e"] == 40.0 and ref_fila["e_prime_median"] == 38.0 and ref_fila["success_pct"] == 100.0
    assert ref_fila["val_over_cv_baseline"] == pytest.approx(40.0 / 46.0)
    assert r["across_new_partitions"]["partitions"] == 2
    assert 0 <= r["across_new_partitions"]["median_e_prime_below_e"] <= 2
    assert r["across_all_including_2026"]["partitions"] == 3
    clave = r["stability"]["2026 (referencia)"]["key_formulas"]
    assert clave["sex·age"]["runs_with_exact_formula"] == 2 and clave["log(1 + |s6|)"]["runs_with_exact_formula"] == 2
    assert r["stability"]["2026 (referencia)"]["variable_use_fraction"]["s6"] == 1.0
    assert r["stability"]["2026 (referencia)"]["variable_use_fraction"]["bmi"] == 0.0
    texto = d23.informe(r)
    assert "Resumen entre particiones" in texto and "Estabilidad" in texto
    # escribir_resumen solo escribe en la base indicada
    import shutil
    destino = tmp_path / "copia_base"
    shutil.copytree(base, destino)
    antes = {p.name for p in destino.iterdir()}
    d23.escribir_resumen(destino, ref)
    assert {p.name for p in destino.iterdir()} - antes == {"resumen.json", "resumen.md"}


def test_el_script_no_nombra_la_lectura_de_validacion():
    assert "load_validation" not in (RAIZ / "scripts" / "run_d23.py").read_text()
    assert "load_validation" not in (RAIZ / "scripts" / "diagnose_tail.py").read_text()


# =============================================================================================
# Diagnóstico de la cola
# =============================================================================================

def test_ultima_mejora_y_generacion_bajo():
    h = [{"best_fitness": x} for x in (1.0, 0.9, 0.9, 0.8, 0.8, 0.8)]
    assert dg.ultima_mejora(h) == 3
    assert dg.ultima_mejora([{"best_fitness": 1.0}] * 4) == 0
    assert dg.generacion_bajo(h, 0.85) == 3 and dg.generacion_bajo(h, 0.1) is None


def test_corte_por_mayor_hueco_ignora_el_atipico_mas_alto():
    v = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 10], dtype=float)
    assert dg.corte_por_mayor_hueco(v) == pytest.approx(1.5)
    assert dg.corte_por_mayor_hueco(v[::-1]) == pytest.approx(1.5)  # no depende del orden


def test_eventos_de_mutacion_aritmetica():
    m = dg.eventos_de_mutacion({"p_mutation": 0.2}, {"head": 7, "p_point": 0.044, "p_is": 0.1, "p_ris": 0.1, "K": 3})
    assert m["tree_per_individual"] == 0.2 and m["gep_gene_length"] == 15
    assert m["gep_per_individual"] == pytest.approx(3 * (0.044 * 15 + 0.2))


def _run(seed, mae, cv, formulas, cplx=10, hist=None):
    return {"meta": {"seed": seed}, "mae": mae, "cv_mae_best": cv, "evolution_best_fitness": cv, "n_columns": 7,
            "kept_originals": ["MedInc", "Latitude"], "formulas": formulas,
            "hall_of_fame": [{"cv_mae_new_folds": cv, "cv_mae_evolution": cv, "complexity": cplx},
                             {"cv_mae_new_folds": cv + 0.001, "cv_mae_evolution": cv - 0.0001, "complexity": cplx + 3}],
            "history": hist or [{"best_fitness": x} for x in (0.54, 0.52, 0.50, 0.49)]}


def test_ficha_usa_el_ganador_del_hall_of_fame():
    f = dg.ficha(_run(3, 0.50, 0.51, ["sqrt(|AveOccup|)"], cplx=12))
    assert f["complexity"] == 12 and f["seed"] == 3 and f["hof_size"] == 2
    assert f["transforms_AveOccup"] is True and f["log_AveRooms"] is False
    assert f["last_improvement_generation"] == 3 and f["generation_below_threshold"] == 2
    assert f["hof_min_cv_evolution"] < f["hof_winner_cv_evolution"]
    assert dg.ficha(_run(4, 0.5, 0.5, ["(AveOccup + MedInc)"]))["transforms_AveOccup"] is False


def test_diagnosticar_separa_grupos_y_marca_las_excepciones(tmp_path):
    rng = np.random.default_rng(0)
    for exp in ("final_C_tree", "final_C_gep"):
        for s in range(20):
            if s < 6:  # malas: CV malo
                mae, cv, fs = 0.52 + 0.001 * s, 0.53, ["(AveOccup + MedInc)"]
            else:  # buenas
                mae, cv, fs = 0.48 + 0.001 * rng.random(), 0.485, ["sqrt(|AveOccup|)", "log(1 + |AveRooms|)"]
            if s == 19:  # excepción: buen CV, mal e′
                mae, fs = 0.66, ["(AveOccup * (AveOccup * Longitude))"]
            f = tmp_path / exp / "california" / f"seed_{s}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(_run(s, mae, cv, fs)))
    cfgs = {"tree": {"p_mutation": 0.2}, "gep": {"head": 7, "p_point": 0.044, "p_is": 0.1, "p_ris": 0.1, "K": 3}}
    d = dg.diagnosticar(tmp_path, cfgs)
    assert d["groups"]["bad"]["n"] == 7 and d["groups"]["good"]["n"] == 13  # 6 malas + el atípico (0.66)
    assert d["groups"]["bad"]["transforms_AveOccup_fraction"] == 0.0 and d["groups"]["good"]["transforms_AveOccup_fraction"] == 1.0
    assert [x["seed"] for x in d["large_gap_runs"]["tree"]] == [19]
    assert d["worst_tree"][0]["seed"] == 19 and len(d["worst_tree"]) == 5 and len(d["best_tree"]) == 5 and len(d["worst_gep"]) == 3
    assert d["spearman_e_prime_vs_cv"]["tree"]["statistic"] > 0.5
    assert d["mutation_events"]["gep_per_individual"] > 10 * d["mutation_events"]["tree_per_individual"]
    texto = dg.informe(d)
    for fragmento in ("Peores, C con árboles", "Hall of fame", "fallo de búsqueda", "Excepciones", "asimetría"):
        assert fragmento in texto
    json.dumps(d, allow_nan=False, default=float)


@pytest.mark.skipif(not (RAIZ / "results" / "runs" / "final_C_tree" / "california").exists(), reason="no hay resultados reales")
def test_diagnostico_sobre_los_resultados_reales():
    d = dg.diagnosticar(RAIZ / "results" / "runs")
    assert d["groups"]["bad"]["n"] + d["groups"]["good"]["n"] == 30
    peor = d["worst_tree"][0]
    assert peor["e_prime"] == max(r["mae"] for r in dg.cargar(RAIZ / "results" / "runs", "final_C_tree"))
    assert d["spearman_e_prime_vs_cv"]["tree"]["statistic"] > 0.9
