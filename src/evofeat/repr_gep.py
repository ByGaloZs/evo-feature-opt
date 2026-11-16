"""Representación GEP: genoma = cabeza + cola, decodificado (Karva) a preorden.

Implementa §8 (variante GEP) y D25 (RF-04, RF-06). Cada genoma es un gen de
``head_len + tail_len`` símbolos; la cola tiene ``t = h·(aridad_máx − 1) + 1`` y solo
terminales, de modo que todo cromosoma decodifica a una fórmula válida de como mucho
``2h + 1`` nodos (h = 7 → 15 nodos, igual que un árbol de profundidad 3).

Operadores (tasas por defecto de Ferreira, §8): mutación puntual 0,044 por símbolo,
transposición IS 0,1, RIS 0,1, recombinación de 1 punto 0,3 y de 2 puntos 0,3. La
recombinación de genes (y, en general, el intercambio de genes entre individuos) vive en
evolution.py (§15.2). Todos los operadores respetan cabeza/cola.

Regla igual que en árboles (D25): la primera posición de la cabeza es siempre una
función, al inicializar y al mutar; así ningún gen es un original sin más.

No hay transposición de gen: OLS no depende del orden de las columnas (§8). Las tasas
de esta clase se leen de la sección ``[gep]`` del TOML; los árboles usan ``p_crossover``
y ``p_mutation``.

Decisiones de detalle confirmadas por el usuario:
- Lo que el fragmento de IS o RIS desplaza fuera de la cabeza se pierde; la cola no se toca.
- Si RIS no encuentra ninguna función, no hace nada.
- Longitud de los fragmentos de IS y RIS: uniforme en 1..3 (valores típicos de Ferreira).
- IS inserta en una posición de la cabeza distinta de la raíz; RIS parte de una función
  de la cabeza (se busca hacia delante desde un punto aleatorio) y se inserta en la raíz.
- Los símbolos de reemplazo de la cabeza son uniformes entre funciones ∪ terminales
  (igual que Grow en árboles).
"""

from __future__ import annotations

import numpy as np

from .formula import Formula
from .primitives import PRIMITIVES

Genome = tuple[str, ...]


def decode_karva(gene: Genome) -> Formula:
    """Decodifica un gen en notación Karva (K-expression) a una fórmula en preorden.

    Los símbolos se leen en orden de aparición, nivel a nivel (anchura): el primero es la
    raíz y cada nodo toma como hijos los siguientes símbolos aún sin asignar, tantos como
    su aridad. Los símbolos que no se alcanzan no se expresan.
    """
    hijos: list[range] = []
    siguiente = 1  # primer símbolo todavía sin padre
    idx = 0
    while idx < siguiente:
        if idx >= len(gene):
            raise ValueError("gen mal formado: faltan símbolos para completar el árbol")
        aridad = PRIMITIVES[gene[idx]][0] if gene[idx] in PRIMITIVES else 0
        hijos.append(range(siguiente, siguiente + aridad))
        siguiente += aridad
        idx += 1
    # recorrido en preorden del árbol construido
    salida: list[str] = []
    pila = [0]
    while pila:
        nodo = pila.pop()
        salida.append(gene[nodo])
        pila.extend(reversed(hijos[nodo]))
    return tuple(salida)


class GepRepresentation:
    """Operadores de GEP sobre un gen cabeza + cola (interfaz ``Representation``, §15.2)."""

    def __init__(
        self,
        n_features: int,
        head_len: int,
        functions: tuple[str, ...] | list[str],
        *,
        p_point: float = 0.044,
        p_is: float = 0.1,
        p_ris: float = 0.1,
        p_one_point: float = 0.3,
        p_two_point: float = 0.3,
        transposon_max_len: int = 3,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features debe ser ≥ 1")
        if head_len < 2:
            raise ValueError("head_len debe ser ≥ 2")
        desconocidas = [f for f in functions if f not in PRIMITIVES]
        if desconocidas or len(functions) == 0:
            raise ValueError(f"conjunto de funciones inválido: {list(functions)}")
        self.n_features = n_features
        self.head_len = head_len
        self.functions = tuple(functions)
        self.terminals = tuple(f"x{i}" for i in range(n_features))
        aridad_max = max(PRIMITIVES[f][0] for f in self.functions)
        self.tail_len = head_len * (aridad_max - 1) + 1
        self.gene_len = head_len + self.tail_len
        self.p_point, self.p_is, self.p_ris = p_point, p_is, p_ris
        self.p_one_point, self.p_two_point = p_one_point, p_two_point
        self.transposon_max_len = min(transposon_max_len, head_len)

    # ---- generación ------------------------------------------------------

    def _terminal(self, rng: np.random.Generator) -> str:
        return self.terminals[int(rng.integers(len(self.terminals)))]

    def _funcion(self, rng: np.random.Generator) -> str:
        return self.functions[int(rng.integers(len(self.functions)))]

    def _simbolo_cabeza(self, rng: np.random.Generator) -> str:
        """Símbolo de la cabeza (no raíz): uniforme entre funciones y terminales."""
        k = int(rng.integers(len(self.functions) + len(self.terminals)))
        return self.functions[k] if k < len(self.functions) else self.terminals[k - len(self.functions)]

    def random_genome(self, rng: np.random.Generator) -> Genome:
        cabeza = [self._funcion(rng)] + [self._simbolo_cabeza(rng) for _ in range(self.head_len - 1)]
        cola = [self._terminal(rng) for _ in range(self.tail_len)]
        return tuple(cabeza + cola)

    def to_formula(self, g: Genome) -> Formula:
        return decode_karva(g)

    # ---- mutación --------------------------------------------------------

    def point_mutation(self, g: Genome, rng: np.random.Generator) -> Genome:
        """Cada símbolo cambia con probabilidad ``p_point``: cabeza → cualquier símbolo
        (la raíz solo una función); cola → solo un terminal."""
        out = list(g)
        for i in np.flatnonzero(rng.random(len(g)) < self.p_point):
            i = int(i)
            if i == 0:
                out[i] = self._funcion(rng)
            elif i < self.head_len:
                out[i] = self._simbolo_cabeza(rng)
            else:
                out[i] = self._terminal(rng)
        return tuple(out)

    def is_transposition(self, g: Genome, rng: np.random.Generator) -> Genome:
        """IS: copia un fragmento del gen y lo inserta en la cabeza (no en la raíz)."""
        largo = int(rng.integers(1, self.transposon_max_len + 1))
        ini = int(rng.integers(0, len(g) - largo + 1))
        fragmento = g[ini : ini + largo]
        destino = int(rng.integers(1, self.head_len))
        cabeza = g[:destino] + fragmento + g[destino : self.head_len]
        return cabeza[: self.head_len] + g[self.head_len :]

    def ris_transposition(self, g: Genome, rng: np.random.Generator) -> Genome:
        """RIS: un fragmento que empieza por una función de la cabeza pasa a la raíz."""
        punto = int(rng.integers(0, self.head_len))
        ini = next((i for i in range(punto, self.head_len) if g[i] in PRIMITIVES), None)
        if ini is None:
            return g  # no hay función desde ese punto: no hay transposición
        largo = int(rng.integers(1, self.transposon_max_len + 1))
        fragmento = g[ini : ini + largo]
        cabeza = fragmento + g[: self.head_len]
        return cabeza[: self.head_len] + g[self.head_len :]

    def mutate(self, g: Genome, rng: np.random.Generator) -> Genome:
        """Mutación puntual (por símbolo) y, con su probabilidad, IS y RIS."""
        g = self.point_mutation(g, rng)
        if rng.random() < self.p_is:
            g = self.is_transposition(g, rng)
        if rng.random() < self.p_ris:
            g = self.ris_transposition(g, rng)
        return g

    # ---- recombinación ---------------------------------------------------

    def one_point_recombination(self, a: Genome, b: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
        """Intercambia todo lo que sigue a un punto de corte (misma posición en ambos)."""
        c = int(rng.integers(1, len(a)))
        return a[:c] + b[c:], b[:c] + a[c:]

    def two_point_recombination(self, a: Genome, b: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
        """Intercambia el tramo comprendido entre dos puntos de corte distintos."""
        c1, c2 = sorted(int(c) for c in rng.choice(len(a) + 1, size=2, replace=False))
        return a[:c1] + b[c1:c2] + a[c2:], b[:c1] + a[c1:c2] + b[c2:]

    def crossover(self, a: Genome, b: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
        """Recombinación de 1 punto y de 2 puntos, cada una con su probabilidad.

        Al intercambiar siempre posiciones homólogas, cabeza y cola conservan su papel.
        """
        if rng.random() < self.p_one_point:
            a, b = self.one_point_recombination(a, b, rng)
        if rng.random() < self.p_two_point:
            a, b = self.two_point_recombination(a, b, rng)
        return a, b
