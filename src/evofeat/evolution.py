"""Bucle evolutivo común a árboles y GEP.

Implementa §8, D7 (hall of fame), D13, D17 (parsimonia lexicográfica), D19 (A/B/C),
D24, D25 (RF-06, RF-07, RF-08, RF-11).

Esquema: generacional con elitismo, torneo con parsimonia lexicográfica, individuos
semilla, operadores de máscara e interruptores idénticos en ambas representaciones y
operadores de fórmulas propios de cada una:

- Árboles: con probabilidad ``p_crossover`` por pareja, cruce uniforme de la máscara y,
  con otro sorteo de la misma probabilidad, sobre UN árbol k al azar, cruce de subárbol o
  intercambio del gen k con su interruptor (50 % cada uno). Con probabilidad ``p_mutation`` por individuo se muta UN árbol al azar.
- GEP: la máscara se cruza como en árboles (``p_crossover``); las tasas de Ferreira se
  aplican de forma independiente a CADA gen (1 punto, 2 puntos, mutación puntual, IS, RIS);
  y con ``p_gene_recombination`` por pareja se intercambia un gen al azar con su interruptor.
- Gen e interruptor son una unidad: el interruptor k viaja con el gen k. Los interruptores
  no se cruzan aparte y mutan por inversión de bit, igual que la máscara (prob. 1/(n+K)).

Configuraciones (D19): A evoluciona solo la máscara (interruptores a 0); B evoluciona solo
los genes con todos los originales fijos y sin límite de columnas (D24 no aplica); C ambos.

Una "generación" es un paso de reproducción: se evalúa la población inicial (generación 0)
y después ``generations`` pasos, cada uno con ``pop_size − elitism`` individuos nuevos. El
presupuesto se mide en individuos generados (100 + 100 · 98 = 9 900), no en evaluaciones
reales: la caché de fitness ahorra evaluaciones pero no individuos.

Toda la aleatoriedad llega por un ``np.random.Generator`` pasado explícitamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .fitness import Fitness
from .individual import Individual, complexity, dropped_constants, phenotype_key, repair

# configuración -> (evoluciona máscara, evoluciona genes, aplica el límite de D24)
CONFIGURACIONES: dict[str, tuple[bool, bool, bool]] = {
    "A": (True, False, True),
    "B": (False, True, False),
    "C": (True, True, True),
}


@dataclass(frozen=True)
class EvolutionParams:
    """Parámetros del bucle (valores de partida del §8)."""

    K: int = 5
    pop_size: int = 100
    generations: int = 100
    tournament_size: int = 3
    elitism: int = 2
    p_crossover: float = 0.8
    p_mutation: float = 0.2
    p_gene_recombination: float = 0.1  # solo GEP
    epsilon: float = 0.0  # parsimonia lexicográfica (D17)
    hof_size: int = 10
    evolve_mask: bool = True
    evolve_genes: bool = True
    enforce_limit: bool = True  # columnas ≤ n − 1 (D24)

    @classmethod
    def for_configuration(cls, nombre: str, **kwargs: Any) -> EvolutionParams:
        """Parámetros de la configuración A, B o C (D19)."""
        if nombre not in CONFIGURACIONES:
            raise ValueError(f"configuración desconocida '{nombre}'; usa A, B o C")
        mascara, genes, limite = CONFIGURACIONES[nombre]
        return cls(evolve_mask=mascara, evolve_genes=genes, enforce_limit=limite, **kwargs)


@dataclass
class EvolutionResult:
    """Resultado de una ejecución del bucle."""

    population: list[Individual]
    fitnesses: list[float]
    best: Individual
    best_fitness: float
    hall_of_fame: list[tuple[float, Individual]]  # mejores fenotipos vistos, de mejor a peor
    history: list[dict[str, float | int]] = field(default_factory=list)
    individuals_generated: int = 0  # presupuesto: individuos creados (iniciales + hijos)


class Evolution:
    """Bucle evolutivo. Una instancia = una ejecución con su ``Fitness`` y su ``rng``."""

    def __init__(
        self, params: EvolutionParams, rep: Any, fitness: Fitness, n: int, rng: np.random.Generator
    ) -> None:
        if params.elitism >= params.pop_size:
            raise ValueError("elitism debe ser menor que pop_size")
        self.p, self.rep, self.fitness, self.n, self.rng = params, rep, fitness, n, rng
        self.ev = fitness.evaluator
        self._hof: dict[str, tuple[float, int, Individual]] = {}
        self._dropped = 0  # constantes descartadas en la generación en curso
        self._generados = 0  # individuos generados (presupuesto)

    # ---- evaluación --------------------------------------------------------

    def _evaluar(self, ind: Individual) -> tuple[float, int]:
        """Fitness y complejidad de un individuo; actualiza el hall of fame y los contadores."""
        self._generados += 1
        valor = self.fitness(ind)
        compl = complexity(ind, self.ev)
        self._dropped += len(dropped_constants(ind, self.ev))
        clave = phenotype_key(ind, self.ev)
        if clave not in self._hof:
            self._hof[clave] = (valor, compl, ind)
            if len(self._hof) > self.p.hof_size:
                peor = max(self._hof, key=lambda k: self._hof[k][:2])
                del self._hof[peor]
        return valor, compl

    def _reparar(self, ind: Individual) -> Individual:
        return repair(ind, self.n, self.rng, self.ev) if self.p.enforce_limit else ind

    # ---- población inicial -------------------------------------------------

    def _inicial(self) -> list[Individual]:
        p, rng, rep, n = self.p, self.rng, self.rep, self.n
        pob: list[Individual] = []
        if p.evolve_mask and p.enforce_limit:
            # D24: una semilla por cada original eliminado, con n − 1 originales y ningún árbol activo
            for j in range(n):
                m = np.ones(n, dtype=bool)
                m[j] = False
                genomas = tuple(rep.random_genome(rng) for _ in range(p.K))
                pob.append(Individual(m, genomas, np.zeros(p.K, dtype=bool)))
        while len(pob) < p.pop_size:
            mascara = rng.random(n) < 0.5 if p.evolve_mask else np.ones(n, dtype=bool)
            genomas = tuple(rep.random_genome(rng) for _ in range(p.K))
            interr = rng.random(p.K) < 0.5 if p.evolve_genes else np.zeros(p.K, dtype=bool)
            pob.append(self._reparar(Individual(mascara, genomas, interr)))
        return pob[: p.pop_size]

    # ---- selección ---------------------------------------------------------

    def _gana(self, a: int, b: int, fit: list[float], comp: list[int]) -> int:
        """Duelo con parsimonia lexicográfica: diferencia < ε → gana el más simple (D17)."""
        if abs(fit[a] - fit[b]) < self.p.epsilon:
            if comp[a] != comp[b]:
                return a if comp[a] < comp[b] else b
        return a if fit[a] <= fit[b] else b

    def _torneo(self, fit: list[float], comp: list[int]) -> int:
        candidatos = self.rng.integers(len(fit), size=self.p.tournament_size)
        campeon = int(candidatos[0])
        for c in candidatos[1:]:
            campeon = self._gana(campeon, int(c), fit, comp)
        return campeon

    # ---- operadores --------------------------------------------------------

    @staticmethod
    def _intercambiar_gen(a: Individual, b: Individual, k: int) -> tuple[Individual, Individual]:
        """Intercambia el gen k y su interruptor entre dos individuos (una unidad)."""
        ga, gb = list(a.genomes), list(b.genomes)
        ga[k], gb[k] = b.genomes[k], a.genomes[k]
        sa, sb = a.switches.copy(), b.switches.copy()
        sa[k], sb[k] = b.switches[k], a.switches[k]
        return replace(a, genomes=tuple(ga), switches=sa), replace(b, genomes=tuple(gb), switches=sb)

    def _cruce(self, a: Individual, b: Individual) -> tuple[Individual, Individual]:
        p, rng, rep = self.p, self.rng, self.rep
        if p.evolve_mask and rng.random() < p.p_crossover:  # cruce uniforme de la máscara
            cambio = rng.random(self.n) < 0.5
            a, b = (
                replace(a, mask=np.where(cambio, b.mask, a.mask)),
                replace(b, mask=np.where(cambio, a.mask, b.mask)),
            )
        if not p.evolve_genes:
            return a, b
        if rep.kind == "tree":
            if rng.random() < p.p_crossover:
                k = int(rng.integers(p.K))
                if rng.random() < 0.5:  # cruce de subárbol entre los árboles de la posición k
                    h1, h2 = rep.crossover(a.genomes[k], b.genomes[k], rng)
                    ga, gb = list(a.genomes), list(b.genomes)
                    ga[k], gb[k] = h1, h2
                    a, b = replace(a, genomes=tuple(ga)), replace(b, genomes=tuple(gb))
                else:  # intercambio de árboles completos
                    a, b = self._intercambiar_gen(a, b, k)
        else:  # GEP: tasas de Ferreira independientes por gen
            ga, gb = list(a.genomes), list(b.genomes)
            for k in range(p.K):
                ga[k], gb[k] = rep.crossover(ga[k], gb[k], rng)
            a, b = replace(a, genomes=tuple(ga)), replace(b, genomes=tuple(gb))
            if rng.random() < p.p_gene_recombination:
                a, b = self._intercambiar_gen(a, b, int(rng.integers(p.K)))
        return a, b

    def _mutacion(self, ind: Individual) -> Individual:
        p, rng, rep = self.p, self.rng, self.rep
        tasa = 1.0 / (self.n + p.K)  # inversión de bit, igual para máscara e interruptores
        mascara, interr = ind.mask, ind.switches
        if p.evolve_mask:
            mascara = mascara ^ (rng.random(self.n) < tasa)
        if p.evolve_genes:
            interr = interr ^ (rng.random(p.K) < tasa)
            genomas = list(ind.genomes)
            if rep.kind == "tree":
                if rng.random() < p.p_mutation:
                    k = int(rng.integers(p.K))
                    genomas[k] = rep.mutate(genomas[k], rng)
            else:
                genomas = [rep.mutate(g, rng) for g in genomas]
            return replace(ind, mask=mascara, genomes=tuple(genomas), switches=interr)
        return replace(ind, mask=mascara, switches=interr)

    # ---- bucle -------------------------------------------------------------

    def _registrar(self, gen: int, pob: list[Individual], fit: list[float], comp: list[int],
                   hits_antes: int, evals_antes: int) -> dict[str, float | int]:
        mejor = min(range(len(pob)), key=lambda i: (fit[i], comp[i]))
        reg = {
            "generation": gen,
            "best_fitness": fit[mejor],
            "median_fitness": float(np.median(fit)),
            "best_complexity": comp[mejor],
            "cache_hits": self.fitness.cache_hits - hits_antes,
            "evaluations": self.fitness.evaluations - evals_antes,
            "constants_dropped": self._dropped,
        }
        self._dropped = 0
        return reg

    def run(self) -> EvolutionResult:
        p = self.p
        history: list[dict[str, float | int]] = []
        hits0, evals0 = self.fitness.cache_hits, self.fitness.evaluations

        pob = self._inicial()
        res = [self._evaluar(i) for i in pob]
        fit, comp = [r[0] for r in res], [r[1] for r in res]
        history.append(self._registrar(0, pob, fit, comp, hits0, evals0))

        for gen in range(1, p.generations + 1):
            hits0, evals0 = self.fitness.cache_hits, self.fitness.evaluations
            orden = sorted(range(len(pob)), key=lambda i: (fit[i], comp[i]))
            nueva = [pob[i] for i in orden[: p.elitism]]
            nfit = [fit[i] for i in orden[: p.elitism]]
            ncomp = [comp[i] for i in orden[: p.elitism]]
            while len(nueva) < p.pop_size:
                a, b = pob[self._torneo(fit, comp)], pob[self._torneo(fit, comp)]
                a, b = self._cruce(a, b)  # cada operador aplica por dentro su propia probabilidad
                for hijo in (a, b):
                    if len(nueva) >= p.pop_size:
                        break
                    hijo = self._reparar(self._mutacion(hijo))
                    f, c = self._evaluar(hijo)
                    nueva.append(hijo)
                    nfit.append(f)
                    ncomp.append(c)
            pob, fit, comp = nueva, nfit, ncomp
            history.append(self._registrar(gen, pob, fit, comp, hits0, evals0))

        mejor = min(range(len(pob)), key=lambda i: (fit[i], comp[i]))
        hof = sorted(self._hof.values(), key=lambda t: (t[0], t[1]))
        return EvolutionResult(
            population=pob,
            fitnesses=fit,
            best=pob[mejor],
            best_fitness=fit[mejor],
            hall_of_fame=[(v, ind) for v, _, ind in hof],
            history=history,
            individuals_generated=self._generados,
        )
