import math
from collections import defaultdict
import random
from itertools import chain, combinations
import networkx as nx
import numpy as np
from qat.linalg import LinAlg
from qat.plugins import ScipyMinimizePlugin
from qat.opt import Ising, CombinatorialProblem, QUBO
from terms import Terms, spin_var, const
from typing import Union


def compute_max_amplitude_M_ij(
    samples: list[tuple[str, float]], terms: Terms, n_samples=None
) -> tuple[int, int]:
    if n_samples is not None:
        reduced_samples = [
            (s, 1 / n_samples)
            for s, _ in random.choices(
                samples, weights=[s[1] for s in samples], k=n_samples
            )
        ]
        samples = reduced_samples

    amplitudes = {}

    var_to_qubit = terms.get_var_to_qubit_mapping()
    qubit_to_var = terms.get_qubit_to_var_mapping()
    for term in terms.terms:
        if len(term) not in [1, 2]:
            continue
        amplitude = 0
        if len(term) == 1:
            i = var_to_qubit[term[0]]
            for bitstring, probability in samples:
                if bitstring[i] == "0":
                    amplitude -= probability
                if bitstring[i] == "1":
                    amplitude += probability
        elif len(term) == 2:
            i, j = var_to_qubit[term[0]], var_to_qubit[term[1]]
            amplitude = 0
            for bitstring, probability in samples:
                if bitstring[i] == bitstring[j]:
                    amplitude += probability
                else:
                    amplitude -= probability
        amplitudes[term] = amplitude

    largest_amplitude = max(amplitudes.values(), key=lambda a: abs(a))

    conclusive_terms = [
        (v, np.sign(a)) for v, a in amplitudes.items() if a == largest_amplitude
    ]
    return random.choice(conclusive_terms)


def rqaoa(
    original_problem: Union[Ising, QUBO, CombinatorialProblem],
    perform_qaoa,
    n_samples=None,
):
    if not isinstance(original_problem, Ising):
        original_problem = original_problem.to_ising()
    J, h, offset = original_problem.get_j_h_and_offset()
    terms = Terms.from_ising(J, h, offset)

    linear_spins = []
    correlations = []

    # Repeat until QUBO has no quadratic constraints
    while terms.get_n_non_trivial_constraints() > 0:
        J, h, offset = terms.to_ising()
        problem = Ising(J, h, offset).to_combinatorial_problem()
        samples = perform_qaoa(problem)
        term, sign = compute_max_amplitude_M_ij(samples, terms, n_samples)
        # only store variable names, not marker that they are spin vars
        if len(term) == 1:
            k = term[0]
            linear_spins.append((k[0], sign))
            terms = terms.substitute(k, const(sign))
        else:
            k, l = term
            correlations.append(((k[0], l[0]), sign))
            terms = terms.substitute(k, spin_var(l, sign))

    # Set remaining linear terms (if there are any).
    for var, c in terms.terms.items():
        if len(var) == 1:
            linear_spins.append((var[0][0], np.sign(c)))

    graph = nx.from_numpy_array(original_problem.j_coupling_matrix)
    solution = compute_final_solution(graph, linear_spins, correlations)
    return [(solution, 1)]


def compute_final_solution(
    graph: nx.Graph,
    linear_spins: list[tuple[int, int]],
    correlations: list[tuple[tuple[int, int], int]],
):
    n = len(graph)
    edges = {v: [] for v in range(n)}
    assignment = {v: None for v in range(n)}

    for (u, v), spin in correlations:
        edges[u].append((v, spin))
        edges[v].append((u, spin))

    def dfs(u):
        for v, spin in edges[u]:
            if assignment[v] is not None:
                continue
            assignment[v] = spin * assignment[u]
            dfs(v)

    for u, sign in linear_spins:
        assignment[u] = sign
        dfs(u)

    for u in range(n):
        if assignment[u] is not None:
            continue
        assignment[u] = random.choice([-1, 1])
        dfs(u)

    return "".join(["1" if assignment[u] == 1 else "0" for u in range(n)])
