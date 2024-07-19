import json
import numpy as np
from parse_params import Params
import statistics
import csv


with open("results/main_evaluation.txt") as file:
    lines = file.readlines()
params = [Params(**json.loads(line)) for line in lines]

approx_times = {}
with open("csvs/classical_approximation_benchmark.csv") as file:
    reader = csv.reader(file)
    next(reader, None)
    for problem, size, seed, runtime in reader:
        size = int(size)
        seed = int(seed)
        runtime = float(runtime)
        approx_times[(problem, size, seed)] = runtime

circuit_optimization_times = {}
with open("csvs/circuit_optimization_benchmarks.csv") as file:
    reader = csv.reader(file)
    next(reader, None)
    for problem, size, n_layers, seed, runtime in reader:
        size = int(size)
        seed = int(seed)
        n_layers = int(n_layers)
        runtime = float(runtime)
        circuit_optimization_times[(problem, size, n_layers, seed)] = runtime

acc_circuit_optimization_times = {}
for (problem, size, n_layers, seed), runtime in circuit_optimization_times.items():
    key = problem, size, n_layers
    if key not in acc_circuit_optimization_times:
        acc_circuit_optimization_times[key] = []
    acc_circuit_optimization_times[key].append(runtime)
avg_circuit_optimization_times = {
    key: statistics.mean(values)
    for key, values in acc_circuit_optimization_times.items()
}


n_samples = 1000
meas_time = 4090  # Median of Qiskit fake backends
nano_seconds = 1 / 1_000_000_000

results = {}
raw_results = {}
performance = {}

for p in params:
    metrics = p.circuit_metrics
    n_shots = p.source.algorithm.n_shots
    problem = p.graphs.problem
    size = p.graphs.size
    n_seeds = p.graphs.n
    n_layers = p.source.algorithm.n_layers
    alg = p.source.algorithm.type

    if not (
        p.source.model is not None
        and p.source.model.noise == 1.0
        and p.source.model.time == 1.0
    ):
        continue

    params_approx_times = []
    params_circuit_optization_times = []
    for seed in range(n_seeds):
        key = problem, size, seed
        runtime = 0
        if key in approx_times and alg in ["wsqaoa", "wsinitqaoa"]:
            runtime = approx_times[key]
        params_approx_times.extend([runtime] * n_shots)

    params_n_circuits = []
    for seed in range(n_seeds):
        key = problem, size, n_layers, seed
        runtime = circuit_optimization_times[key]
        n_circuits = [len(c) for c in metrics[seed * n_shots : (seed + 1) * n_shots]]

        for inst_n_circuits in n_circuits:
            circuit_runtime = runtime
            for i in range(inst_n_circuits - 1):
                circuit_size = size - 1 - i
                circuit_runtime += avg_circuit_optimization_times[
                    (problem, size, n_layers)
                ]
            params_circuit_optization_times.append(circuit_runtime)

    optimizer_time = [sum(c.optimizer_time for c in inst) for inst in metrics]
    optimizer_iterations = [
        sum(c.n_optimizer_iterations for c in inst) for inst in metrics
    ]
    circuit_depth = [sum(c.depth for c in inst) for inst in metrics]
    n_circuits = [len(inst) for inst in metrics]
    circuit_time = [
        sum(
            (c.depth + meas_time) * n_samples * c.n_optimizer_iterations * nano_seconds
            for c in inst
        )
        for inst in metrics
    ]
    total_time = [
        sum(v)
        for v in zip(
            optimizer_time,
            circuit_time,
            params_approx_times,
            params_circuit_optization_times,
        )
    ]

    key = (
        p.graphs.problem,
        p.graphs.size,
        p.source.algorithm.type,
        p.source.algorithm.n_layers,
    )
    results[key] = statistics.mean(total_time)
    raw_results[key] = total_time
    performance[key] = p.raw


problems = [
    ("maxcut", "Max-Cut"),
    ("partition", "Partition"),
    ("vertex_cover", "Vertex Cover"),
]
algorithms = [
    ("qaoa", "QAOA"),
    ("wsqaoa", "WSQAOA"),
    ("wsinitqaoa", "WS-Init-QAOA"),
    ("rqaoa", "RQAOA"),
]


def to_dict(l):
    return [a for a, _ in l], {a: b for a, b in l}


problems, problem_names = to_dict(problems)
algorithms, algorithm_names = to_dict(algorithms)

n_layers = list(range(1, 4 + 1))
sizes = list(range(5, 10 + 1))

with open("csvs/performance_by_runtime.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        ("runtime", "performance", "problem", "n_qubits", "algorithm", "n_qaoa_layers")
    )
    for problem in problems:
        for algorithm in algorithms:
            for size in sizes:
                for layer in n_layers:
                    key = (
                        problem,
                        size,
                        algorithm,
                        layer,
                    )
                    for p, r in zip(performance[key], raw_results[key]):
                        writer.writerow(
                            (
                                r,
                                p,
                                problem_names[problem],
                                size,
                                algorithm_names[algorithm],
                                layer,
                            )
                        )
