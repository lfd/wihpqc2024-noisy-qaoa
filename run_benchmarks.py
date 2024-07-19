from collections import defaultdict
from itertools import chain, combinations, product
import math
import argparse
import warnings
import json5 as json
import time
import random
from typing import Callable, NewType
import traceback
import multiprocessing
import cProfile
import pstats

import networkx as nx
import numpy as np
from cvxgraphalgs.algorithms.max_cut import _solve_cut_vector_program, _recover_cut

from qat.core import Job, Circuit
from qat.core.qpu import CommonQPU
from qat.qpus import LinAlg, NoisyQProc
from qat.plugins import ScipyMinimizePlugin

# from qlmaas.plugins import ScipyMinimizePlugin as QlmaasScipyMinimizePlugin  # type: ignore
from qat.hardware import DefaultHardwareModel
from qat.quops import (
    ParametricAmplitudeDamping,
    ParametricPureDephasing,
    QuantumChannelKraus,
    make_depolarizing_channel,
)
from qat.quops.metrics import get_average_process_fidelity
from qat.core.util import get_syntax
from qat.lang.AQASM import RY, RZ, AbstractGate
from qat.opt import Ising, MaxCut, NumberPartitioning
from qat.noisy import compute_fidelity


from generate_param_sets import spread_json, check_for_duplicates
from parse_params import *
from transpile_to_sx_rz import transpile as transpile_to_sx_rz
from rqaoa import rqaoa

##
## Optimization problems
##


def non_empty_subsets(s: list) -> chain:
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def max_cut_size(graph: nx.Graph) -> int:
    nodes = [n for n in graph]
    size = -math.inf
    for s in non_empty_subsets(nodes):
        size = max(size, nx.cut_size(graph, s))
    return size


def compute_number_partitioning_size(numbers: list[float], spins: list[int]) -> float:
    return sum(numbers) - abs(sum((n * s for n, s in zip(numbers, spins))))


def best_number_partitioning_size(numbers: list[float]) -> float:
    all_spins = product([1, -1], repeat=len(numbers))
    best = -np.inf
    for spins in all_spins:
        result = compute_number_partitioning_size(numbers, spins)
        best = max(result, best)
    return best


def compute_vertex_cover_size(graph: nx.Graph, spins):
    # check if vertices not in the solution form an independent set
    subgraph = graph.subgraph([i for i, s in enumerate(spins) if s == -1])
    is_vertex_cover = subgraph.number_of_edges() == 0
    if is_vertex_cover:
        return 1 / (len(graph) - subgraph.number_of_nodes())
    else:
        return 1 / len(graph)


def best_vertex_cover_size(graph: nx.Graph):
    complement = nx.complement(graph)
    maximum_independent_set = max(len(c) for c in nx.find_cliques(complement))
    minimum_vertex_cover = graph.number_of_nodes() - maximum_independent_set
    return 1 / minimum_vertex_cover


def compute_optimal_solution(problem: Ising) -> float:
    if isinstance(problem, MaxCut):
        return max_cut_size(problem.graph)
    if isinstance(problem, NumberPartitioning):
        return best_number_partitioning_size(problem.array_of_numbers)
    if isinstance(problem, VertexCover):
        return best_vertex_cover_size(problem.graph)
    raise ValueError(f"Cannot compute optimal solution for problem {problem}")


def evaluate_solution(bitstring: str, problem: Ising) -> float:
    if isinstance(problem, MaxCut):
        indices_bin_0 = np.where(np.array(list(bitstring), dtype=int) == 0)[0]
        return nx.cut_size(problem.graph, indices_bin_0)
    if isinstance(problem, NumberPartitioning):
        numbers = problem.array_of_numbers
        spins = [1 if b == "0" else -1 for b in bitstring]
        return compute_number_partitioning_size(numbers, spins)
    if isinstance(problem, VertexCover):
        spins = [-1 if b == "0" else 1 for b in bitstring]
        graph = problem.graph
        return compute_vertex_cover_size(graph, spins)
    raise ValueError(f"Cannot evaluate solution for problem {problem}")


Samples = NewType("Samples", list[tuple[str, float]])


def compute_success_prob(samples: Samples, problem: Ising, max_size: float) -> float:
    prob_success = 0
    for bitstring, probability in samples:
        if evaluate_solution(bitstring, problem) == max_size:
            prob_success += probability
    return prob_success


def compute_approximation_ratio(
    samples: Samples, problem: Ising, max_size: float
) -> float:
    expected_cut_size = 0
    for bitstring, probability in samples:
        expected_cut_size += evaluate_solution(bitstring, problem) * probability
    return expected_cut_size / max_size


def generate_uniform_solution(n: int) -> Samples:
    def gen_bitstrings(n: int):
        if n > 0:
            yield from (
                bits + bit for bits in gen_bitstrings(n - 1) for bit in ("0", "1")
            )
        else:
            yield ""

    bitstrings = list(gen_bitstrings(n))
    prob = 1 / len(bitstrings)
    return [(b, prob) for b in bitstrings]


# cf. https://github.com/hermish/cvx-graph-algorithms
def perform_maxcut_sdp(graph: nx.Graph) -> np.ndarray:
    adjacency = nx.to_scipy_sparse_array(graph).toarray()
    return _solve_cut_vector_program(adjacency)


def sample_gw_rounding(solution: np.ndarray, n=1000) -> Samples:
    results = defaultdict(int)
    for _ in range(n):
        vector = apply_gw_rounding(solution)
        bitstring = "".join([str(i) for i in vector])
        results[bitstring] += 1

    return [(bitstring, count / n) for bitstring, count in results.items()]


def apply_gw_rounding(solution: np.ndarray) -> np.ndarray:
    sides = _recover_cut(solution)
    solution = 0.5 * (1 + np.real(sides))
    return np.rint(solution).astype(int)


def partition_greedy(numbers: list[float]) -> list[int]:
    bits = []
    s1 = 0
    s2 = 0
    for n in numbers:
        if s1 <= s2:
            bits.append(0)
            s1 += n
        else:
            bits.append(1)
            s2 += n
    return bits


def vertex_cover_approx(graph: nx.Graph) -> list[int]:
    edges = list(graph.edges)
    cover = []
    while len(edges) > 0:
        u, v = random.choice(edges)
        cover.append(u)
        cover.append(v)
        edges = [(i, j) for i, j in edges if i not in [u, v] and j not in [u, v]]
    result = [0] * len(graph)
    for v in cover:
        result[v] = 1
    return result


def gen_problem(graphs: Graphs, seed: int) -> Ising:
    if graphs.problem in ["maxcut", "vertex_cover"]:
        graph = nx.generators.random_graphs.erdos_renyi_graph(
            graphs.size, 0.5, seed=seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if graphs.problem == "maxcut":
                return MaxCut(graph)
            else:
                return VertexCover(graph)
    if graphs.problem == "partition":
        np.random.seed(seed)
        numbers = np.random.rand(graphs.size)
        return NumberPartitioning(numbers)
    raise ValueError(f"Unsupported problem {graphs.problem}")


def create_approx_solution(problem: Ising) -> list[int]:
    if isinstance(problem, MaxCut):
        sdp_solution = perform_maxcut_sdp(problem.graph)
        return apply_gw_rounding(sdp_solution)
    elif isinstance(problem, NumberPartitioning):
        return partition_greedy(problem.array_of_numbers)
    elif isinstance(problem, VertexCover):
        return vertex_cover_approx(problem.graph)
    raise ValueError(f"No approximation algorithm available for problem {problem}")


##
## Noise model
##

Readout = AbstractGate("Readout", [], arity=1, matrix_generator=lambda: np.identity(2))
two_qubit_gates = {"CNOT"}


def gate_n_qubits(gate: str) -> int:
    return 2 if gate in two_qubit_gates else 1


class NoiseParameters:
    def __init__(
        self,
        t1: float,
        t2: float,
        gate_errors: dict[str, float],
        gate_times: dict[str, float],
        prep0_meas1: float,
        prep1_meas0: float,
    ):
        self.t1 = t1
        self.t2 = t2
        self.gate_errors = gate_errors
        self.gate_times = gate_times
        self.prep0_meas1 = prep0_meas1
        self.prep1_meas0 = prep1_meas0


noise_parameters = {
    "qiskit_average": NoiseParameters(
        t1=100000,
        t2=100000,
        gate_errors={
            "H": 0.001,
            "PH": 0.001,
            "RX": 0.001,
            "RY": 0.001,
            "RZ": 0.001,
            "SX": 0.0004,
            "CNOT": 0.03,
        },
        gate_times={
            "H": 50,
            "PH": 50,
            "RX": 50,
            "RY": 50,
            "RZ": 50,
            "SX": 50,
            "CNOT": 600,
        },
        prep0_meas1=0.028,
        prep1_meas0=0.050,
    ),
    "qiskit_median": NoiseParameters(
        t1=100000,
        t2=85000,
        gate_errors={
            "H": 0.0005,
            "PH": 0,
            "RX": 0.0005,
            "RY": 0.001,
            "RZ": 0,
            "WM": 0.001,
            "SX": 0.0003,
            "CNOT": 0.01,
        },
        gate_times={
            "H": 45,
            "PH": 0,
            "RX": 45,
            "RY": 90,
            "RZ": 0,
            "WM": 90,
            "SX": 35,
            "CNOT": 400,
        },
        prep0_meas1=0.014,
        prep1_meas0=0.033,
    ),
    "qiskit_best": NoiseParameters(
        t1=600000,
        t2=495000,
        gate_errors={
            "H": 0.0004,
            "PH": 0,
            "RX": 0.0004,
            "RY": 0.0009,
            "RZ": 0,
            "WM": 0.0009,
            "SX": 0.0002,
            "CNOT": 0.006,
        },
        gate_times={
            "H": 35,
            "PH": 0,
            "RX": 35,
            "RY": 70,
            "RZ": 0,
            "WM": 70,
            "SX": 20,
            "CNOT": 270,
        },
        prep0_meas1=0.0016,
        prep1_meas0=0.0010,
    ),
}


def compose_channels_sequential(
    a: QuantumChannelKraus, b: QuantumChannelKraus
) -> QuantumChannelKraus:
    out_ops = []
    for a_op in a.kraus_operators:
        for b_op in b.kraus_operators:
            out_ops.append(b_op @ a_op)
    return QuantumChannelKraus(out_ops)


def compose_channels_parallel(
    a: QuantumChannelKraus, b: QuantumChannelKraus
) -> QuantumChannelKraus:
    out_ops = []
    for a_op in a.kraus_operators:
        for b_op in b.kraus_operators:
            out_ops.append(np.kron(a_op, b_op))
    return QuantumChannelKraus(out_ops)


def compose_parametric_channels(
    channels: list[Callable[[float], QuantumChannelKraus]],
    gate_time: float,
    n_qubits: int,
) -> QuantumChannelKraus:
    one_bit_channel = channels[0](gate_time)
    for c in channels[1:]:
        one_bit_channel = compose_channels_sequential(one_bit_channel, c(gate_time))

    channel = one_bit_channel
    for _ in range(1, n_qubits):
        channel = compose_channels_parallel(channel, one_bit_channel)
    return channel


def create_parametric_channel_by_fidelity(
    channels: list[Callable[[float], QuantumChannelKraus]],
    target_fidelity: float,
    n_qubits: int,
    start=50,
    acc=0.00001,
) -> QuantumChannelKraus:
    current = start
    while True:
        channel = compose_parametric_channels(channels, current, n_qubits)
        fidelity = get_average_process_fidelity(channel)
        if fidelity > target_fidelity:
            current *= 2
        else:
            break

    low = 0
    high = current

    while abs(fidelity - target_fidelity) > acc:
        middle = (low + high) / 2
        channel = compose_parametric_channels(channels, middle, n_qubits)
        fidelity = get_average_process_fidelity(channel)
        if fidelity > target_fidelity:
            low = middle
        else:
            high = middle

    return channel


def create_composite_model_from_parameters(
    params: NoiseParameters, gate_error_factor=1, gate_time_factor=1, readout_error=0.0
) -> DefaultHardwareModel:
    """
    Main noise model used for the noisy QAOA benchmarks.
    Gate noise is represented by the sequential composition of thermal relaxation and depolarizing noise.

    Args:
        params (NoiseParameters): Noise parameters
        gate_error_factor (int, optional): Scales gate depolarizing probabilities. Defaults to 1.0.
        gate_time_factor (int, optional): Scales gate durations. Defaults to 1.0.
        readout_error (float, optional): Scales readout error probabilities. Defaults to 0.0.

    Returns:
        DefaultHardwareModel: Noise model
    """
    scaled_gate_times = {
        gate: time * gate_time_factor for gate, time in params.gate_times.items()
    }
    # 1 / T_2 = 1 / (2 * T_1) + 1 / T_phi
    t_phi = 1 / (1 / params.t2 - 1 / (2 * params.t1))
    amplitude_damping = ParametricAmplitudeDamping(T_1=params.t1)
    dephasing = ParametricPureDephasing(T_phi=t_phi)
    idle_noise = [amplitude_damping, dephasing]

    def compute_depolarizing_prob(gate):
        # model: gate infidelity comes from thermal relaxation + depolarizing
        # compute depolarizing probability accordingly
        # (based on original time and fidelity!)
        infidelity = params.gate_errors[gate]
        gate_time = params.gate_times[gate]
        n_qubits = gate_n_qubits(gate)
        if gate_time > 0:
            idle_channel = compose_parametric_channels(idle_noise, gate_time, n_qubits)
        else:
            idle_channel = QuantumChannelKraus([np.identity(2**n_qubits)])
        idle_fidelity = get_average_process_fidelity(idle_channel)
        target_fidelity = 1 - infidelity
        depolarizing_prob = (idle_fidelity - target_fidelity) / (
            idle_fidelity - 2 ** (-n_qubits)
        )
        return depolarizing_prob

    def generate_readout_noise(prep0_meas1, prep1_meas0):
        channel = QuantumChannelKraus(
            [
                np.sqrt(prep0_meas1) * np.array([[0, 0], [1, 0]]),
                np.array(
                    [[np.sqrt(1 - prep0_meas1), 0], [0, np.sqrt(1 - prep1_meas0)]]
                ),
                np.sqrt(prep1_meas0) * np.array([[0, 1], [0, 0]]),
            ]
        )
        return lambda *_: channel

    def generate_gate_noise(gate):
        if gate == Readout.name:
            return generate_readout_noise(
                params.prep0_meas1 * readout_error, params.prep1_meas0 * readout_error
            )
        n_qubits = gate_n_qubits(gate)

        identity = QuantumChannelKraus([np.identity(2**n_qubits)])
        gate_time = scaled_gate_times[gate]
        idle_channel = (
            identity
            if gate_time == 0
            else compose_parametric_channels(idle_noise, gate_time, n_qubits)
        )

        init_depol_prob = compute_depolarizing_prob(gate)
        depol_prob = init_depol_prob * gate_error_factor
        # print("depol", gate, init_depol_prob, depol_prob)

        depol_channel = (
            identity
            if depol_prob == 0
            else make_depolarizing_channel(
                depol_prob,
                nqbits=n_qubits,
                method_2q="equal_probs",
                depol_type="randomizing",
            )
        )

        gate_channel = compose_channels_sequential(idle_channel, depol_channel)
        infidelity = params.gate_errors[gate]
        # print(gate, 1 - infidelity, get_average_process_fidelity(gate_channel))

        # if number of Kraus operators is larger than what is theoretically necessary,
        if gate_channel.kraus_rank() > 2 ** (2 * n_qubits):
            gate_channel = gate_channel.to_choi().to_kraus()

        return lambda *_: gate_channel

    gates = list(params.gate_errors.keys())
    if readout_error > 0.0:
        gates.append(Readout.name)
    gate_noise = {g: generate_gate_noise(g) for g in gates}

    return DefaultHardwareModel(
        gate_times=scaled_gate_times, gate_noise=gate_noise, idle_noise=idle_noise
    )


def create_fidelity_model_from_parameters(
    params: NoiseParameters, gate_error_factor=1, time=None, readout_error=0.0
) -> DefaultHardwareModel:
    """
    Noise model which models gate errors solely as depolarizing noise.

    Args:
        params (NoiseParameters): Noise parameters.
        gate_error_factor (int, optional): Scales depolarizing probabilities. Defaults to 1.
        time (float, optional): Ignored
        readout_error (float, optional): Ignored

    Returns:
        DefaultHardwareModel: Noise model
    """

    def generate_gate_noise(gate, error_prob):
        n_qubits = 2 if gate in two_qubit_gates else 1
        infidelity = error_prob * gate_error_factor
        fidelity = 1 - infidelity
        p = (1 - fidelity) / (1 - 2 ** (-n_qubits))
        channel = make_depolarizing_channel(
            p, nqbits=n_qubits, depol_type="randomizing", method_2q="equal_probs"
        )
        return lambda *_: channel

    gate_noise = {g: generate_gate_noise(g, p) for g, p in params.gate_errors.items()}
    t_phi = 1 / (1 / params.t2 - 1 / (2 * params.t1))

    return DefaultHardwareModel(
        gate_times=params.gate_times,
        gate_noise=gate_noise,
        idle_noise=[
            ParametricAmplitudeDamping(T_1=params.t1),
            ParametricPureDephasing(T_phi=t_phi),
        ],
    )


class ThermalRelaxationByDepolarizing:
    def __init__(self, T_1, T_2):
        self.T_1 = T_1
        self.T_2 = T_2
        self.T_phi = 1 / (1 / T_2 - 1 / (2 * T_1))

    def __call__(self, t):
        fidelity = 1 / 2 + 1 / 6 * np.exp(-t / self.T_1) + 1 / 3 * np.exp(-t / self.T_2)
        depol_prob = (1 - fidelity) / (1 - 2 ** (-1))
        return make_depolarizing_channel(
            depol_prob, nqbits=1, depol_type="randomizing", method_2q="equal_probs"
        )


def create_depolarizing_model_from_parameters(
    params: NoiseParameters, gate_error_factor=1, time=None, readout_error=0.0
) -> DefaultHardwareModel:
    """
    Noise model representing gate errors and thermal relaxation as depolarizing channels with the same fidelity.

    Args:
        params (NoiseParameters): _description_
        gate_error_factor (int, optional): Scales depolarizing probabilities. Defaults to 1.
        time (_type_, optional): Ignored.
        readout_error (float, optional): Ignored.

    Returns:
        DefaultHardwareModel: Noise model
    """

    def generate_gate_noise(gate, error_prob):
        n_qubits = 2 if gate in two_qubit_gates else 1
        infidelity = error_prob * gate_error_factor
        fidelity = 1 - infidelity
        p = (1 - fidelity) / (1 - 2 ** (-n_qubits))
        channel = make_depolarizing_channel(
            p, nqbits=n_qubits, depol_type="randomizing", method_2q="equal_probs"
        )
        return lambda *_: channel

    gate_noise = {g: generate_gate_noise(g, p) for g, p in params.gate_errors.items()}
    # 1 / T_2 = 1 / (2 * T_1) + 1 / T_phi
    t_phi = 1 / (1 / params.t2 - 1 / (2 * params.t1))

    return DefaultHardwareModel(
        gate_times=params.gate_times,
        gate_noise=gate_noise,
        idle_noise=[ThermalRelaxationByDepolarizing(params.t1, params.t2)],
    )


noisy_models = {
    "noisy_fidelity": create_fidelity_model_from_parameters,
    "noisy_composite": create_composite_model_from_parameters,
    "noisy_depolarizing": create_depolarizing_model_from_parameters,
}


##
## QAOA
##


def wsqaoa_mixer(theta: float, beta: float) -> np.ndarray:
    return (
        RY.matrix_generator(-theta)
        @ RZ.matrix_generator(-2 * beta)
        @ RY.matrix_generator(theta)
    )


WM = AbstractGate("WM", [float, float], arity=1, matrix_generator=wsqaoa_mixer)


def create_wsqaoa_ansatz(
    ansatz: Job,
    solution: list[int],
    epsilon: float,
    wsinitqaoa=False,
    custom_gate=True,
) -> Job:
    def solution_bit_to_theta(bit):
        if bit <= epsilon:
            return 2 * math.asin(math.sqrt(epsilon))
        if bit >= 1 - epsilon:
            return 2 * math.asin(math.sqrt(1 - epsilon))
        return 2 * math.asin(math.sqrt(bit))

    thetas = [solution_bit_to_theta(bit) for bit in solution]

    circuit = ansatz.circuit

    ops = circuit.ops
    index = 0
    while index < len(ops):
        name, variables, qubits = get_syntax(circuit, index)
        # Initialize differently
        if name == "H":
            qubit = qubits[0]
            ops.pop(index)
            circuit.insert_gate(index, RY(thetas[qubit]), qubits)
        # WS-Init-QAOA uses the original mixer Hamiltonian
        if name == "RX" and not wsinitqaoa:
            qubit = qubits[0]
            beta = variables[0]
            theta = thetas[qubit]
            ops.pop(index)
            if custom_gate:
                circuit.insert_gate(index, WM(theta, beta), qubits)
            else:
                circuit.insert_gate(index, RY(theta), qubits)
                circuit.insert_gate(index + 1, RZ(-2 * beta), qubits)
                circuit.insert_gate(index + 2, RY(-theta), qubits)
                index += 2
        index += 1
    return ansatz


def append_readout_gates(circuit: Circuit):
    for i in range(circuit.nbqbits):
        circuit.insert_gate(len(circuit.ops) + i, Readout(), [i])


class ProfilerMinimizePlugin(ScipyMinimizePlugin):
    def set_profiler(self, arr):
        self.profiler = arr

    def evaluate(self, *args, **xargs):
        self.profiler.disable()
        result = super().evaluate(*args, **xargs)
        self.profiler.enable()
        return result

    def optimize(self, *args, **xargs):
        # print("(optimize")
        self.profiler.enable()
        result = super().optimize(*args, **xargs)
        return result


def perform_qaoa(
    ansatz: Job, qpu: CommonQPU, use_qlmaas: bool, metric: str = ""
) -> Samples:
    minimizer_fn = QlmaasScipyMinimizePlugin if use_qlmaas else ProfilerMinimizePlugin  # type: ignore
    minimizer = minimizer_fn(
        method="COBYLA",
        tol=1e-2,
        options={"maxiter": 150},
    )
    if not use_qlmaas:
        profiler = cProfile.Profile()
        minimizer.set_profiler(profiler)
    stack = minimizer | qpu()

    optimizer_time = None

    if use_qlmaas:
        async_result = stack.submit(ansatz)
        result = async_result.join()
    else:
        result = stack.submit(ansatz)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        for func, data in list(stats.stats.items())[:10]:
            file = func[0]
            function = func[2]
            if "cobyla" in file and function == "calcfc":
                optimizer_time = data[3]
                break

    params = eval(result.meta_data["parameters"])

    sol_job = ansatz(
        **{key: var for key, var in zip(ansatz.get_variables(), params)}
    ).circuit.to_job()

    n_optimizer_iterations = len(json.loads(result.meta_data["optimization_trace"]))
    depth = sol_job.circuit.depth(
        gate_times=noise_parameters["qiskit_median"].gate_times
    )
    n_qubits = sol_job.circuit.nbqbits
    circuit_metrics = CircuitMetrics(
        n_qubits, depth, n_optimizer_iterations, optimizer_time
    )

    if metric == "fidelity":
        fidelity, _ = compute_fidelity(sol_job.circuit, qpu())
        return fidelity, circuit_metrics

    if use_qlmaas:
        async_solution_result = qpu().submit(sol_job)
        solution_result = async_solution_result.join()
    else:
        solution_result = qpu().submit(sol_job)

    output = [(s.state.bitstring, s.probability) for s in solution_result]
    if use_qlmaas:
        async_result.delete_files()
        async_solution_result.delete_files()
    return output, circuit_metrics


def create_qpu(source: Source, conn) -> CommonQPU:  # type: ignore
    if source.model.type == "ideal":

        def qpu():
            if conn is None:
                return LinAlg()
            return conn.get_qpu("qat.qpus:LinAlg")()

    elif source.model.type in noisy_models:
        hardware_model = noisy_models[source.model.type](
            noise_parameters[source.model.params],
            source.model.noise,
            source.model.time,
            source.model.readout,
        )
        n_samples = source.model.n_samples
        if n_samples is not None:
            sim_method = "stochastic"
        else:
            sim_method = "deterministic-vectorized"

        def qpu():
            if conn is None:
                noisy_qpu = NoisyQProc
            else:
                noisy_qpu = conn.get_qpu("qat.qpus:NoisyQProc")
            return noisy_qpu(
                hardware_model=hardware_model,
                n_samples=n_samples,
                sim_method=sim_method,
            )

    else:
        raise ValueError(f"Model {source.model} not supported")
    return qpu


def execute_model(source: Source, problem: Ising, conn, metric: str) -> Samples:  # type: ignore
    use_qlmaas = conn is not None
    if source.type == "qpu":
        qpu = create_qpu(source, conn)
        return execute_quantum_model(
            qpu,
            source.algorithm,
            source.model.sx,
            source.model.optimize_circuit_depth,
            source.model.readout,
            problem,
            use_qlmaas,
            metric,
        )
    if source.type == "random":
        if not isinstance(problem, Ising):
            problem = problem.to_ising()
        return generate_uniform_solution(len(problem.magnetic_field_h)), None
    if source.type == "gw_rounding":
        if isinstance(problem, MaxCut):
            return sample_gw_rounding(perform_maxcut_sdp(problem.graph), n=10000), None
        if isinstance(problem, NumberPartitioning):
            greedy_result = partition_greedy(problem.array_of_numbers)
            bitstring = "".join([str(i) for i in greedy_result])
            return [(bitstring, 1)], None
        if isinstance(problem, VertexCover):
            solutions = {}
            for _ in range(1000):
                greedy_result = vertex_cover_approx(problem.graph)
                bitstring = "".join([str(i) for i in greedy_result])
                if bitstring not in solutions:
                    solutions[bitstring] = 0
                solutions[bitstring] += 1
            total = sum((n for n in solutions.values()))
            return [(bitstring, n / total) for bitstring, n in solutions.items()], None
        raise ValueError(f"GW rounding mode not supported for problem {problem}")


def execute_quantum_model(
    qpu: CommonQPU,
    algorithm: Algorithm,
    sx_transpilation: bool,
    circuit_optimization: bool,
    readout: float,
    problem: Ising,
    use_qlmaas: bool,
    metric: str,
) -> Samples:
    algorithm_type = algorithm.type
    n_layers = algorithm.n_layers

    def create_ansatz(problem):
        if circuit_optimization:
            return create_optimized_qaoa_circuit(problem, n_layers)
        else:
            return problem.qaoa_ansatz(n_layers)

    def transpile(circuit):
        if sx_transpilation:
            transpile_to_sx_rz(circuit)
        if readout > 0.0:
            append_readout_gates(circuit)

    if algorithm_type == "rqaoa":
        circuit_metrics = []

        def rqaoa_perform_qaoa(problem):
            ansatz = create_ansatz(problem)
            transpile(ansatz.circuit)
            result, metrics = perform_qaoa(ansatz, qpu, use_qlmaas)
            circuit_metrics.append(metrics)
            return result

        result = rqaoa(problem, rqaoa_perform_qaoa, algorithm.n_samples)
        return result, circuit_metrics

    if algorithm_type == "qaoa":
        ansatz = create_ansatz(problem)
    elif algorithm_type == "wsqaoa" or algorithm_type == "wsinitqaoa":
        wsinitqaoa = algorithm_type == "wsinitqaoa"
        approx = create_approx_solution(problem)
        qaoa_ansatz = create_ansatz(problem)
        ansatz = create_wsqaoa_ansatz(
            qaoa_ansatz,
            approx,
            epsilon=0.25,
            wsinitqaoa=wsinitqaoa,
        )
    transpile(ansatz.circuit)
    result, metrics = perform_qaoa(ansatz, qpu, use_qlmaas, metric)
    return result, [metrics]


def worker(
    seed: int, params: Params, use_qlmaas: bool, verbose: bool
) -> tuple[float, list[CircuitMetrics]]:
    conn = None
    if use_qlmaas:
        conn = QLMaaSConnection("localhost", check_host=False)  # type: ignore

    n_tries = 0
    while True:
        try:
            problem = gen_problem(params.graphs, seed)
            samples, metrics = execute_model(
                params.source, problem, conn, params.metric
            )

            max_size = compute_optimal_solution(problem)
            if params.metric == "success_prob":
                result = compute_success_prob(samples, problem, max_size)
            elif params.metric == "approx_ratio":
                result = compute_approximation_ratio(samples, problem, max_size)
            elif params.metric in ["fidelity", "running_time"]:
                result = samples
            else:
                raise ValueError(f"Unknown metric type {params.metric}")
            if verbose:
                print(f"{seed}: {result}")
            return result, metrics

        except Exception:
            n_tries += 1
            print(f"seed {seed}, retry {n_tries}")
            traceback.print_exc()
            time.sleep(10 + random.random() * 50)


def average_results(values):
    assert len(values) > 0

    total = values[0]
    if type(total) is not tuple:
        total = (total,)
    for value in values[1:]:
        if type(value) is not tuple:
            value = (value,)

        total = tuple([a + b for a, b in zip(total, value)])

    avg = tuple([a / len(values) for a in total])
    if len(avg) == 1:
        avg = avg[0]
    return avg


def main(
    param_sets: list[Params],
    meta: str,
    verbose: bool,
    n_threads: int,
    out: str,
    qlmaas: bool,
    raw: bool,
    store_circuit_metrics: bool,
    done_marker: bool,
):
    meta = json.loads(meta)

    pool = multiprocessing.Pool(n_threads)

    total_results = []

    with pool:
        for params in param_sets:
            n_shots = 1
            if params.source.algorithm is not None:
                n_shots = params.source.algorithm.n_shots
            results = [
                [
                    pool.apply_async(worker, (s, params, qlmaas, verbose))
                    for _ in range(n_shots)
                ]
                for s in range(params.graphs.n)
            ]
            total_results.append(results)

        for results_by_graph, params in zip(total_results, param_sets):
            results = []
            metrics = []
            for seed, graph_result in enumerate(results_by_graph):
                raw_graph_result = [r.get() for r in graph_result]
                n_shots = len(graph_result)
                graph_result = [r[0] for r in raw_graph_result if r[0] is not None]
                graph_metrics = [r[1] for r in raw_graph_result]
                if len(graph_result) != n_shots:
                    print("not all shots completed")
                avg = average_results(graph_result)
                if verbose and n_shots > 1:
                    print(f"{seed} avg: {avg}")
                results.append(avg)
                metrics.extend(graph_metrics)

            if len(results) != params.graphs.n:
                print("completed", len(results), "graphs")

            params.value = average_results(results)
            if raw:
                params.raw = results
            if store_circuit_metrics:
                params.circuit_metrics = metrics
            params.time = int(time.time())
            params.meta = meta
            print(params.dump_json(), flush=True)
            if out is not None:
                with open(out + ".out", "a+") as file:
                    file.write(params.dump_json() + "\n")

    if done_marker:
        with open(out + ".out", "r") as file:
            content = file.read()
        with open(out + ".txt", "w") as file:
            file.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        "-p",
        required=True,
        help="Benchmark parameters (#qubits, algorithm, ...) as JSON5 string (cf. parse_params.py)",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Enable verbose output (print result for every algorithm run)",
    )
    parser.add_argument(
        "--meta",
        "-m",
        default="{}",
        help="JSON5 string which is attached to the benchmark results for later filtering",
    )
    parser.add_argument(
        "--n_threads",
        "-P",
        type=int,
        default=1,
        help="Degree of parallelism for the simulations",
    )
    parser.add_argument(
        "--out", "-o", help="Path to the output file (without *.out file extension)"
    )
    parser.add_argument(
        "--qlmaas", action="store_true", help="Use QLMaaS (experimental)"
    )
    parser.add_argument(
        "--raw",
        "-r",
        action="store_true",
        help="Add results for individual instances to output (not just average)",
    )
    parser.add_argument(
        "--done_marker",
        action="store_true",
        help="Copy benchmarks results into *.txt file when they are finished",
    )
    parser.add_argument(
        "--circuit_metrics",
        "-c",
        action="store_true",
        help="Log metrics like circuit depth, n iterations, optimizer time etc.",
    )

    args = parser.parse_args()

    param_sets = [
        json.loads(Params(**p).dump_json())
        for p in spread_json(json.loads(args.params))
    ]
    param_sets = check_for_duplicates(param_sets, args.out + ".out")

    param_sets = [Params(**p) for p in param_sets]
    print(f"Run {len(param_sets)} evaluations")

    main(
        param_sets,
        args.meta,
        args.v,
        args.n_threads,
        args.out,
        args.qlmaas,
        args.raw,
        args.done_marker,
    )
