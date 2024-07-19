from typing import Union
import numpy as np


class Terms:
    def __init__(self, terms: dict = None):
        if terms is None:
            terms = {}
        self.terms = terms
        self.qubit_mapping = None

    def clean_up(self):
        to_delete = []
        for var, cof in self.terms.items():
            if cof == 0:
                to_delete.append(var)
        for var in to_delete:
            del self.terms[var]
        return self

    def insert(self, var, coef=None):
        var = tuple(sorted(list(var)))
        self.qubit_mapping = None
        if var not in self.terms:
            self.terms[var] = 0
        self.terms[var] += coef
        if self.terms[var] == 0:
            del self.terms[var]

    def scale(self, scalar: float) -> "Terms":
        result = {}
        if scalar == 0:
            return Terms(result)
        for var, coef in self.terms.items():
            result[var] = coef * scalar
        return Terms(result)

    def add(self, other: "Terms") -> "Terms":
        return add_terms([self, other])

    def add_in_place(self, other: "Terms") -> "Terms":
        for var, coef in other.terms.items():
            self.insert(var, coef)

    def __add__(self, other: "Terms") -> "Terms":
        return self.add(other)

    def __sub__(self, other: "Terms") -> "Terms":
        return self.add(other.scale(-1))

    def __neg__(self) -> "Terms":
        return self.scale(-1)

    def mul(self, other: "Terms") -> "Terms":
        result = Terms()
        for var1, coef1 in self.terms.items():
            for var2, coef2 in other.terms.items():
                var = combine_keys(var1, var2)
                result.insert(var, coef1 * coef2)
        return result

    def __mul__(self, other: "Terms") -> "Terms":
        return self.mul(other)

    def substitute(self, var_to_sub: Union["Terms", tuple], terms: "Terms") -> "Terms":
        if isinstance(var_to_sub, Terms):
            assert len(var_to_sub.terms) == 1
            var_to_sub = next(iter(var_to_sub.terms))
            assert len(var_to_sub) == 1
            var_to_sub = var_to_sub[0]

        result = Terms()

        for var, coef in self.terms.items():
            if var_to_sub in var:
                var_set = set(var)
                var_set.remove(var_to_sub)
                new_var = tuple(var_set)
                result.add_in_place(Terms({new_var: coef}) * terms)
            else:
                result.insert(var, coef)
        return result

    def to_binary(self):
        result = Terms()
        for var, coef in self.terms.items():
            spin_vars = [v for v in var if v[1] == True]
            t = Terms({var: coef})
            for v in spin_vars:
                t = t.substitute(v, Terms({((v[0], False),): 2, (): -1}))
            result.add_in_place(t)
        return result

    def to_spin(self):
        result = Terms()
        for var, coef in self.terms.items():
            binary_vars = [v for v in var if v[1] == False]
            t = Terms({var: coef})
            for v in binary_vars:
                t = t.substitute(v, Terms({((v[0], True),): 0.5, (): 0.5}))
            result.add_in_place(t)
        return result

    def _get_qubit_mapping(self):
        if self.qubit_mapping is not None:
            return self.qubit_mapping
        terms_vars = set()
        for var_tuple in self.terms:
            for var in var_tuple:
                terms_vars.add(var)
        sorted_vars = sorted(list(terms_vars))
        var_to_qubit = {}
        qubit_to_var = {}
        for i, var in enumerate(sorted_vars):
            var_to_qubit[var] = i
            qubit_to_var[i] = var
        self.qubit_mapping = var_to_qubit, qubit_to_var
        return self.qubit_mapping

    def get_var_to_qubit_mapping(self):
        return self._get_qubit_mapping()[0]

    def get_qubit_to_var_mapping(self):
        return self._get_qubit_mapping()[1]

    def get_n_vars(self):
        return len(self.get_var_to_qubit_mapping())

    def get_n_non_trivial_constraints(self):
        return len([var for var in self.terms if len(var) > 1])

    def to_qubo(self):
        terms = self.to_binary()

        var_to_index = terms.get_var_to_qubit_mapping()
        n = len(var_to_index)
        Q = np.zeros((n, n))

        for var, coef in terms.terms.items():
            if var == ():
                continue
            if len(var) == 1:
                i = var_to_index[var[0]]
                Q[i, i] = coef
            elif len(var) == 2:
                var1 = var[0]
                var2 = var[1]
                coef_half = coef / 2
                i = var_to_index[var1]
                j = var_to_index[var2]
                Q[i, j] = Q[j, i] = coef_half
            else:
                raise Exception("to_qubo can only handle terms with degree at most 2")

        offset = terms.terms[()] if () in terms.terms else 0
        return Q, offset

    @staticmethod
    def from_qubo(Q, offset, index_to_var: dict) -> "Terms":
        n = len(Q)
        assert Q.shape == (n, n)
        if index_to_var is None:
            index_to_var = {i: (i, False) for i in range(n)}

        for i in index_to_var:
            if type(index_to_var[i]) is not tuple:
                index_to_var[i] = (index_to_var[i], False)

        for v, b in index_to_var.values():
            assert b == False, "index_to_var in from_qubo must map to binary variables"

        terms = Terms()
        for i in range(n):
            for j in range(n):
                terms.insert((index_to_var[i], index_to_var[j]), Q[i, j])
        terms.insert((), offset)
        return terms

    def to_ising(self):
        terms = self.to_spin()
        var_to_index = terms.get_var_to_qubit_mapping()

        n = len(var_to_index)
        J = np.zeros((n, n))
        h = np.zeros(n)

        for var, coef in terms.terms.items():
            if var == ():
                continue
            if len(var) == 1:
                i = var_to_index[var[0]]
                h[i] = coef
            elif len(var) == 2:
                var1, var2 = var
                coef_half = coef / 2
                i = var_to_index[var1]
                j = var_to_index[var2]
                J[i, j] = J[j, i] = coef_half
            else:
                raise Exception("to_qubo can only handle terms with degree at most 2")

        offset = terms.terms[()] if () in terms.terms else 0
        return J, h, offset

    @staticmethod
    def from_ising(J, h, offset, index_to_var: dict = None) -> "Terms":
        n = len(h)
        assert J.shape == (n, n)
        assert h.shape == (n,)
        if index_to_var is None:
            index_to_var = {i: (i, True) for i in range(n)}

        for i in index_to_var:
            if type(index_to_var[i]) is not tuple:
                index_to_var[i] = (index_to_var[i], True)

        for v, b in index_to_var.values():
            assert b == True, "index_to_var in from_ising must map to spin variables"

        terms = Terms()
        for i in range(n):
            terms.insert((index_to_var[i],), h[i])

        for i in range(n):
            for j in range(n):
                terms.insert((index_to_var[i], index_to_var[j]), J[i, j])

        terms.insert((), offset)
        return terms

    def __str__(self):
        return str(self.terms)


def combine_keys(a, b):
    binary_vars = []
    spin_vars = []
    a = list(a)
    b = list(b)

    for v, is_spin in a + b:
        if is_spin:
            spin_vars.append(v)
        else:
            binary_vars.append(v)

    simplified_binary_vars = list(set(binary_vars))

    spin_var_counts = {}
    for v in spin_vars:
        if v not in spin_var_counts:
            spin_var_counts[v] = 0
        spin_var_counts[v] += 1

    simplified_spin_vars = [v for v, count in spin_var_counts.items() if count % 2 == 1]

    return tuple(
        sorted(
            [(v, False) for v in simplified_binary_vars]
            + [(v, True) for v in simplified_spin_vars]
        )
    )


def binary_var(name, factor=1):
    if type(name) is tuple:
        name = name[0]
    return Terms({((name, False),): factor})


def spin_var(name, factor=1):
    if type(name) is tuple:
        name = name[0]
    return Terms({((name, True),): factor})


def const(factor=1):
    return Terms({(): factor})


def add_terms(terms_list: list[Terms]):
    result = Terms()
    for terms in terms_list:
        for var, coef in terms.terms.items():
            result.insert(var, coef)
    return result
