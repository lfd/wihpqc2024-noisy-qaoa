import math
from qat.core import Circuit
from qat.core.util import get_syntax
from qat.lang.AQASM import RZ, RX, RY, AbstractGate
import numpy as np
from qat.core.variables import ArithExpression

ws_mixer_angles = """
+ - angle * + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5 angle * + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5 3.141592653589793
+ UMINUS * 2 atan2 abs * + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5 abs * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5 3.141592653589793
+ angle * + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5 angle * + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 ** - * + * cos / UMINUS \\theta 2 cos / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 + * sin / UMINUS \\theta 2 UMINUS sin / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 * + * cos / UMINUS \\theta 2 UMINUS sin / \\theta 2 * UMINUS sin / UMINUS \\theta 2 * exp * 1j * -2 \\beta cos / \\theta 2 + * sin / UMINUS \\theta 2 cos / \\theta 2 * cos / UMINUS \\theta 2 * exp * 1j * -2 \\beta sin / \\theta 2 -0.5
""".strip().split(
    "\n"
)


def wsqaoa_mixer(theta: float, beta: float) -> np.ndarray:
    return (
        RY.matrix_generator(-theta)
        @ RZ.matrix_generator(-2 * beta)
        @ RY.matrix_generator(theta)
    )


def sqrt_x() -> np.ndarray:
    return RX.matrix_generator(math.pi / 2)


WM = AbstractGate("WM", [float, float], arity=1, matrix_generator=wsqaoa_mixer)
SX = AbstractGate("SX", [], arity=1, matrix_generator=sqrt_x)


def hadardmard() -> list[np.ndarray]:
    return [SX(), RZ(math.pi / 2), SX()]


def rx(angle) -> list[np.ndarray]:
    return [RZ(-math.pi / 2), SX(), RZ(math.pi - angle), SX(), RZ(-math.pi / 2)]


def u(theta: float, phi: float, gamma: float) -> list[np.ndarray]:
    return [
        RZ(gamma - math.pi / 2),
        SX(),
        RZ(math.pi - theta),
        SX(),
        RZ(phi - math.pi / 2),
    ]


def ry(angle: float) -> list[np.ndarray]:
    return [
        RZ(-math.pi),
        SX(),
        RZ(math.pi - angle),
        SX(),
        RZ(0),
    ]


def wm(theta: float, beta: float) -> list[np.ndarray]:
    arith_expressions = []
    for expr in ws_mixer_angles:
        expr = expr.replace("\\theta", str(theta))
        expr = expr.replace("\\beta", str(beta))
        arith_expression = ArithExpression.from_string(expr)
        arith_expressions.append(arith_expression)
        # arith_expression.pretty_print()
    return [
        RZ(arith_expressions[0]),
        SX(),
        RZ(arith_expressions[1]),
        SX(),
        RZ(arith_expressions[2]),
    ]


def wm_oj(theta: float, beta: float) -> list[np.ndarray]:
    return [
        RY(theta),
        RZ(-2 * beta),
        RY(-theta),
    ]


def transpile(circuit: Circuit):
    ops = circuit.ops
    index = 0

    def replace_gate(new_gates):
        nonlocal index
        ops.pop(index)
        for i, gate in enumerate(new_gates):
            circuit.insert_gate(index + i, gate, qubits)
        index += len(gates) - 1

    while index < len(ops):
        name, variables, qubits = get_syntax(circuit, index)
        if name == "H":
            gates = hadardmard()
            replace_gate(gates)
        if name == "RX":
            angle = variables[0]
            gates = rx(angle)
            replace_gate(gates)
        if name == "RY":
            angle = variables[0]
            gates = ry(angle)
            replace_gate(gates)
        if name == "WM":
            theta, beta = variables
            gates = wm(theta, beta)
            replace_gate(gates)
        index += 1


# def execute(job):
#     qpu = LinAlg()
#     result = qpu.submit(job)

#     results = []
#     for sample in result:
#         results.append(sample.probability)
#     return results


# import random

# from qat.opt.max_cut import MaxCut
# import networkx as nx
# from run_benchmarks import create_wsqaoa_ansatz


# def main():
#     for seed in range(11):
#         size = 6
#         n_layers = 2
#         graph = nx.generators.random_graphs.erdos_renyi_graph(size, 1.5, seed=seed)
#         problem = MaxCut(graph)
#         # ansatz = problem.qaoa_ansatz(n_layers)
#         ansatz = create_wsqaoa_ansatz(
#             graph,
#             n_layers=n_layers,
#             solution=[random.choice([1, 1]) for _ in range(size)],
#             epsilon=random.random() * 1.5,
#         )
#         bindings = {
#             var: random.random() * 3 * math.pi for var in ansatz.get_variables()
#         }
#         qpu = LinAlg()
#         value2 = qpu.submit(ansatz(**bindings)).value
#         value3 = qpu.submit(ansatz(**bindings)).value
#         transpile(ansatz.circuit)
#         print(math.isclose(value2, value2), value1, value2)

#     # job.circuit.display()

#     # print(H.matrix_generator())


# if __name__ == "__main__":
#     main()
