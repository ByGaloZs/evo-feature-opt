"""Transformador evolutivo P con interfaz fit / transform.

Implementa D2, D3, D7, D8, D9 (RF-03, RF-08): ``fit(Xt, yt)`` aprende y
``transform(X)`` solo aplica, de modo que la fuga es imposible por construcción.

``fit`` hace, usando únicamente (Xt, yt):
1. Imputación por mediana (D3) y estandarización de los originales (D2, antes de la síntesis).
2. El bucle evolutivo con folds fijos ``KFold(5, shuffle=True, random_state=semilla)``.
3. La re-selección del ganador entre el hall of fame con CV repetida (3 × 5-fold) sobre
   particiones nuevas, ``random_state = semilla + 10 000 + r`` (D7).
4. Guarda el fenotipo final: la máscara y la lista de fórmulas efectivas (D14). ``transform``
   las aplica tal cual, sin recalcular constancia ni volver a ajustar nada.

El escalado de Xt′ (D2, después) no vive aquí: va dentro del pipeline de M (D8).

La configuración ``baseline`` no evoluciona: devuelve los originales (imputados y escalados).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .evolution import Evolution, EvolutionParams
from .fitness import Fitness
from .formula import Formula, to_infix
from .individual import (
    Individual,
    active_formulas,
    complexity,
    dropped_constants,
    transform_matrix,
)
from .models import make_model
from .repr_gep import GepRepresentation
from .repr_tree import TreeRepresentation

FUNCIONES_POR_DEFECTO = ("add", "sub", "mul", "div", "sqrt", "log")
CONFIGURACIONES_VALIDAS = ("baseline", "A", "B", "C")
REPRESENTACIONES_VALIDAS = ("tree", "gep")


@dataclass(frozen=True)
class RunConfig:
    """Configuración de un experimento (§15.3). Valores por defecto = valores de partida."""

    # [experiment]
    name: str = "run"
    datasets: tuple[str, ...] = ("diabetes", "california")
    configuration: str = "C"  # baseline | A | B | C
    representation: str = "tree"  # tree | gep
    seed_start: int = 0
    seed_count: int = 30
    # [representation]
    K: int = 5
    max_depth: int = 3
    gep_head: int | None = None  # None -> 2^max_depth − 1 (D20: 2 -> 3, 3 -> 7, 4 -> 15)
    functions: tuple[str, ...] = FUNCIONES_POR_DEFECTO
    # [evolution]
    pop_size: int = 100
    generations: int = 100
    tournament_size: int = 3
    elitism: int = 2
    p_crossover: float = 0.8
    p_mutation: float = 0.2
    # [gep]
    p_point: float = 0.044
    p_is: float = 0.1
    p_ris: float = 0.1
    p_one_point: float = 0.3
    p_two_point: float = 0.3
    p_gene_recombination: float = 0.1
    transposon_max_len: int = 3
    # [fitness]
    model: str = "ols"
    cv_folds: int = 5
    epsilon: float = 0.0
    max_cached_values: int = 1000
    # [final_selection]
    hof_size: int = 10
    repeats: int = 3

    def __post_init__(self) -> None:
        if self.configuration not in CONFIGURACIONES_VALIDAS:
            raise ValueError(f"configuration debe ser una de {CONFIGURACIONES_VALIDAS}")
        if self.representation not in REPRESENTACIONES_VALIDAS:
            raise ValueError(f"representation debe ser una de {REPRESENTACIONES_VALIDAS}")

    @property
    def head(self) -> int:
        """Longitud de la cabeza de GEP (h), deducida de la profundidad si no se fija (D20)."""
        return self.gep_head if self.gep_head is not None else 2**self.max_depth - 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        """Construye la configuración desde un diccionario con las secciones del TOML.

        Falla ante secciones o claves desconocidas (evita typos que cambien resultados).
        """
        secciones = {"experiment", "representation", "evolution", "gep", "fitness", "final_selection"}
        sobran = set(d) - secciones
        if sobran:
            raise ValueError(f"secciones desconocidas en la configuración: {sorted(sobran)}")
        validos = {f.name for f in fields(cls)}
        plano: dict[str, Any] = {}
        for seccion, claves in d.items():
            for clave, valor in claves.items():
                if seccion == "experiment" and clave == "seeds":
                    plano["seed_start"], plano["seed_count"] = int(valor["start"]), int(valor["count"])
                    continue
                if seccion == "experiment" and clave == "configuration":
                    plano["configuration"] = valor
                    continue
                if seccion == "evolution" and clave == "max_columns":
                    if valor != "n-1":
                        raise ValueError('max_columns solo admite "n-1" (D24)')
                    continue
                if seccion == "final_selection" and clave in ("hof_size", "repeats"):
                    plano[clave] = valor
                    continue
                if clave not in validos:
                    raise ValueError(f"clave desconocida [{seccion}] {clave}")
                plano[clave] = tuple(valor) if isinstance(valor, list) else valor
        return cls(**plano)


def load_config(ruta: str) -> RunConfig:
    """Lee un archivo TOML de ``configs/`` (RNF-01)."""
    with open(ruta, "rb") as f:
        return RunConfig.from_dict(tomllib.load(f))


@dataclass
class _Estado:
    """Todo lo que ``fit`` aprende; nada más se usa en ``transform``."""

    names: list[str]
    mask: np.ndarray
    formulas: list[Formula]


class EvoFeatureTransformer:
    """Transformador P: sintetiza atributos con PG/GEP multiárbol y selecciona originales."""

    def __init__(self, cfg: RunConfig, seed: int) -> None:
        self.cfg, self.seed = cfg, seed
        self._estado: _Estado | None = None
        self.best_: Individual | None = None
        self.formulas_: list[str] = []
        self.history_: list[dict[str, float | int]] = []
        self.kept_originals_: list[str] = []
        self.constant_formulas_dropped_: list[str] = []
        self.cv_mae_best_: float = float("nan")  # MAE en CV repetida con folds nuevos (D7)
        self.evolution_best_fitness_: float = float("nan")  # MAE con los folds de la evolución
        self.cv_fold_maes_: list[float] = []  # MAE por fold del ganador (folds de la evolución)
        self.complexity_: int = 0  # columnas + nodos efectivos del ganador (D17)
        self.individuals_generated_: int = 0
        self.distinct_evaluations_: int = 0
        self.cache_hits_: int = 0
        self.hall_of_fame_: list[dict[str, Any]] = []

    # ---- ayudas ------------------------------------------------------------

    def _preparar(self, X: np.ndarray) -> np.ndarray:
        """Imputación por mediana y estandarización con los parámetros aprendidos en train."""
        return self.scaler_.transform(self.imputer_.transform(np.asarray(X, dtype=np.float64)))

    def _crear_representacion(self, n: int) -> Any:
        c = self.cfg
        if c.representation == "tree":
            return TreeRepresentation(n, c.max_depth, c.functions)
        return GepRepresentation(
            n, c.head, c.functions,
            p_point=c.p_point, p_is=c.p_is, p_ris=c.p_ris,
            p_one_point=c.p_one_point, p_two_point=c.p_two_point,
            transposon_max_len=c.transposon_max_len,
        )

    def _parametros(self) -> EvolutionParams:
        c = self.cfg
        return EvolutionParams.for_configuration(
            c.configuration, K=c.K, pop_size=c.pop_size, generations=c.generations,
            tournament_size=c.tournament_size, elitism=c.elitism, p_crossover=c.p_crossover,
            p_mutation=c.p_mutation, p_gene_recombination=c.p_gene_recombination,
            epsilon=c.epsilon, hof_size=c.hof_size,
        )

    # ---- fit ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> EvoFeatureTransformer:
        """Aprende la transformación usando solo (Xt, yt)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[1]
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(n)]
        if len(names) != n:
            raise ValueError("feature_names debe tener una entrada por columna")
        self.imputer_ = SimpleImputer(strategy="median").fit(X)  # D3
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))  # D2 (antes)
        Xs = self._preparar(X)

        if self.cfg.configuration == "baseline":
            self._estado = _Estado(names, np.ones(n, dtype=bool), [])
            self.best_ = Individual(np.ones(n, dtype=bool), (), np.zeros(0, dtype=bool))
            self._registrar_fenotipo(names)
            self.cv_mae_best_ = self._cv_baseline(Xs, y)
            return self

        c = self.cfg
        rep = self._crear_representacion(n)
        folds = list(KFold(c.cv_folds, shuffle=True, random_state=self.seed).split(Xs))
        fit_fn = Fitness(Xs, y, folds, lambda: make_model(c.model, self.seed), rep, c.max_cached_values)
        res = Evolution(self._parametros(), rep, fit_fn, n, np.random.default_rng(self.seed)).run()
        self.history_ = res.history
        self.individuals_generated_ = res.individuals_generated
        self.distinct_evaluations_ = fit_fn.evaluations
        self.cache_hits_ = fit_fn.cache_hits

        # D7: re-selección entre el hall of fame con CV repetida sobre particiones nuevas
        folds_nuevos = [
            list(KFold(c.cv_folds, shuffle=True, random_state=self.seed + 10_000 + r).split(Xs))
            for r in range(c.repeats)
        ]
        candidatos = []
        for valor_evol, ind in res.hall_of_fame:
            cv = float(np.mean([fit_fn.cv_mae(ind, f) for f in folds_nuevos]))
            candidatos.append((cv, complexity(ind, fit_fn.evaluator), valor_evol, ind))
        self.hall_of_fame_ = [
            {"cv_mae_new_folds": cv, "cv_mae_evolution": ve, "complexity": co}
            for cv, co, ve, _ in sorted(candidatos, key=lambda t: (t[0], t[1]))
        ]
        cv, _, ve, ganador = min(candidatos, key=lambda t: (t[0], t[1]))
        self.best_ = ganador
        self.cv_mae_best_, self.evolution_best_fitness_ = cv, ve
        ev = fit_fn.evaluator
        efectivas = active_formulas(ganador, ev)
        self._estado = _Estado(names, ganador.mask.copy(), efectivas)
        self._registrar_fenotipo(names)
        self.constant_formulas_dropped_ = [to_infix(f, names) for f in dropped_constants(ganador, ev)]
        self.cv_fold_maes_ = fit_fn.fold_maes(ganador)
        self.complexity_ = complexity(ganador, ev)
        return self

    def _registrar_fenotipo(self, names: list[str]) -> None:
        est = self._estado
        assert est is not None
        self.kept_originals_ = [nm for nm, m in zip(names, est.mask) if m]
        self.formulas_ = [to_infix(f, names) for f in est.formulas]

    def _cv_baseline(self, Xs: np.ndarray, y: np.ndarray) -> float:
        """CV repetida del modelo con los originales (referencia para ``baseline``)."""
        c = self.cfg
        rep = TreeRepresentation(Xs.shape[1], c.max_depth, c.functions)
        fit_fn = Fitness(Xs, y, [], lambda: make_model(c.model, self.seed), rep, c.max_cached_values)
        ind = self.best_
        assert ind is not None
        folds = [
            list(KFold(c.cv_folds, shuffle=True, random_state=self.seed + 10_000 + r).split(Xs))
            for r in range(c.repeats)
        ]
        return float(np.mean([fit_fn.cv_mae(ind, f) for f in folds]))

    # ---- transform ---------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la transformación aprendida; no reajusta nada (D9)."""
        if self._estado is None:
            raise RuntimeError("el transformador no está ajustado; llama a fit primero")
        return transform_matrix(self._estado.mask, self._estado.formulas, self._preparar(X))

    @property
    def n_columns(self) -> int:
        """Columnas de Xt′ (efectivas)."""
        if self._estado is None:
            raise RuntimeError("el transformador no está ajustado; llama a fit primero")
        return int(self._estado.mask.sum()) + len(self._estado.formulas)
