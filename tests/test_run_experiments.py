"""Tests del hito 9: configuraciones finales y lanzador (RF-10, D19, D25)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evofeat.experiment import config_hash, ruta_resultado
from evofeat.transformer import load_config

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_experiments", RAIZ / "scripts" / "run_experiments.py")
lanzador = importlib.util.module_from_spec(_spec)
sys.modules["run_experiments"] = lanzador  # necesario para que los procesos hijos puedan importar las funciones
_spec.loader.exec_module(lanzador)

FINALES = {
    "final_baseline": ("baseline", "tree"),
    "final_A": ("A", "tree"),
    "final_B": ("B", "tree"),
    "final_C_tree": ("C", "tree"),
    "final_C_gep": ("C", "gep"),
}


# --- configuraciones finales del repositorio -----------------------------------------------------

@pytest.mark.parametrize("nombre", sorted(FINALES))
def test_config_final(nombre):
    c = load_config(str(RAIZ / "configs" / f"{nombre}.toml"))
    conf, rep = FINALES[nombre]
    assert (c.name, c.configuration, c.representation) == (nombre, conf, rep)
    assert c.datasets == ("diabetes", "california")
    assert (c.seed_start, c.seed_count) == (0, 30)  # semillas finales 0–29 (§15.5)
    assert (c.K, c.max_depth, c.head) == (3, 3, 7)  # piloto (D20)
    assert c.epsilon_for("diabetes") == 1.26 and c.epsilon_for("california") == 0.003
    assert (c.pop_size, c.generations, c.tournament_size, c.elitism) == (100, 100, 3, 2)
    assert (c.p_crossover, c.p_mutation) == (0.8, 0.2)
    assert (c.model, c.cv_folds, c.hof_size, c.repeats) == ("ols", 5, 10, 3)


def test_las_configs_solo_difieren_en_configuracion_y_representacion():
    """D25: C-árbol y C-GEP cambian solo la representación; el resto es idéntico."""
    from dataclasses import replace
    cs = {n: load_config(str(RAIZ / "configs" / f"{n}.toml")) for n in FINALES}
    ref = cs["final_C_tree"]
    for nombre, c in cs.items():
        assert replace(c, name=ref.name, configuration=ref.configuration, representation=ref.representation) == ref, nombre
    assert cs["final_C_gep"].configuration == ref.configuration


def test_no_hay_otros_toml_ni_nombres_repetidos():
    archivos = sorted(p.stem for p in (RAIZ / "configs").glob("*.toml"))
    assert archivos == sorted(FINALES)


def test_planificacion_completa_300_ejecuciones(tmp_path):
    rutas = [RAIZ / "configs" / f"{n}.toml" for n in sorted(FINALES)]
    tareas = lanzador.planificar(rutas, tmp_path)
    assert len(tareas) == 5 * 2 * 30  # 5 experimentos × 2 datasets × 30 semillas
    # las más largas primero: California con B/C antes que Diabetes
    assert tareas[0][1] == "california" and tareas[0][0].configuration in ("B", "C")
    assert tareas[-1][1] == "diabetes"
    assert {(t[0].name, t[1], t[2]) for t in tareas} == {
        (n, d, s) for n in FINALES for d in ("diabetes", "california") for s in range(30)}


# --- lanzador con presupuesto pequeño --------------------------------------------------------------

def _toml_pequeno(tmp_path: Path, nombre="mini", extra="") -> Path:
    ruta = tmp_path / f"{nombre}.toml"
    ruta.write_text(f"""
[experiment]
name = "{nombre}"
datasets = ["diabetes"]
configuration = "C"
representation = "tree"
seeds = {{ start = 0, count = 3 }}
[evolution]
pop_size = 12
generations = 2
[final_selection]
hof_size = 3
repeats = 1
{extra}
""")
    return ruta


def test_lanzar_ejecuta_todo_y_relanzar_no_repite(tmp_path):
    cfg = _toml_pequeno(tmp_path)
    runs = tmp_path / "runs"
    r1 = lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    assert (r1["pendientes"], r1["hechas"], r1["fallos"]) == (3, 3, [])
    c = load_config(str(cfg))
    archivos = [ruta_resultado(c, "diabetes", s, runs) for s in range(3)]
    assert all(a.exists() for a in archivos)
    antes = [a.read_bytes() for a in archivos]
    r2 = lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    assert (r2["pendientes"], r2["hechas"]) == (0, 0)
    assert [a.read_bytes() for a in archivos] == antes  # ni se reescriben


def test_si_falta_una_semilla_solo_se_ejecuta_esa(tmp_path):
    cfg = _toml_pequeno(tmp_path)
    runs = tmp_path / "runs"
    lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    c = load_config(str(cfg))
    ruta_resultado(c, "diabetes", 1, runs).unlink()
    r = lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    assert (r["pendientes"], r["hechas"]) == (1, 1)
    assert ruta_resultado(c, "diabetes", 1, runs).exists()


def test_json_guardado_lleva_el_hash_del_toml(tmp_path):
    cfg = _toml_pequeno(tmp_path)
    runs = tmp_path / "runs"
    lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    reg = json.loads(ruta_resultado(load_config(str(cfg)), "diabetes", 0, runs).read_text())
    assert reg["meta"]["config_hash"] == config_hash(cfg)


def test_toml_cambiado_detiene_el_lanzador(tmp_path):
    cfg = _toml_pequeno(tmp_path)
    runs = tmp_path / "runs"
    lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    cfg.write_text(cfg.read_text() + "\n# cambio\n")  # mismo experimento, otro hash
    with pytest.raises(RuntimeError, match="otro TOML"):
        lanzador.planificar([cfg], runs)


def test_nombres_repetidos_fallan(tmp_path):
    a = _toml_pequeno(tmp_path, "uno")
    b = tmp_path / "otro.toml"
    b.write_text(a.read_text())  # mismo name = "uno"
    with pytest.raises(ValueError, match="repetidos"):
        lanzador.planificar([a, b], tmp_path / "runs")


def test_un_fallo_no_detiene_el_resto_y_se_reintenta(tmp_path, monkeypatch):
    cfg = _toml_pequeno(tmp_path)
    runs = tmp_path / "runs"
    original = lanzador.run_and_save

    def rota(cfg_, dataset, seed, h, dir_runs):
        if seed == 1:
            raise RuntimeError("fallo simulado")
        return original(cfg_, dataset, seed, h, dir_runs)

    monkeypatch.setattr(lanzador, "run_and_save", rota)
    r = lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    assert r["hechas"] == 2 and len(r["fallos"]) == 1 and r["fallos"][0][2] == 1
    c = load_config(str(cfg))
    assert not ruta_resultado(c, "diabetes", 1, runs).exists()  # sin JSON a medias
    monkeypatch.setattr(lanzador, "run_and_save", original)
    r2 = lanzador.lanzar([cfg], jobs=1, dir_runs=runs, verbose=False)
    assert (r2["pendientes"], r2["hechas"]) == (1, 1)  # solo la que falló


def test_en_paralelo_da_los_mismos_resultados_que_en_serie(tmp_path):
    cfg = _toml_pequeno(tmp_path)
    c = load_config(str(cfg))
    lanzador.lanzar([cfg], jobs=1, dir_runs=tmp_path / "serie", verbose=False)
    r = lanzador.lanzar([cfg], jobs=2, dir_runs=tmp_path / "par", verbose=False)
    assert r["fallos"] == [] and r["hechas"] == 3
    for s in range(3):
        a = json.loads(ruta_resultado(c, "diabetes", s, tmp_path / "serie").read_text())
        b = json.loads(ruta_resultado(c, "diabetes", s, tmp_path / "par").read_text())
        for campo in ("mae", "baseline_mae", "formulas", "kept_originals", "cv_mae_best", "history",
                      "distinct_evaluations", "secondary"):
            assert a[campo] == b[campo], campo


def test_nucleos_fisicos_razonable():
    assert 1 <= lanzador.nucleos_fisicos() <= 256
