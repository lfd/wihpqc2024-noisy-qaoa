import csv
from parse_params import Params
import json
from filter_results import apply_filter


files = [
    "results/main_evaluation.txt",
    "results/bounds.txt",
]

params = []

for file_name in files:
    with open(file_name) as file:
        lines = file.readlines()
    params.extend([Params(**json.loads(line)) for line in lines])

data_points = {}
for p in params:
    model = p.source.model
    if not (
        model is not None
        and (
            model.type == "ideal"
            or (
                model.type == "noisy_composite"
                and model.noise == 1.0
                and model.time == 1.0
            )
        )
    ):
        continue
    problem = p.graphs.problem
    n = p.graphs.size
    alg = p.source.algorithm.type
    depth = p.source.algorithm.n_layers
    noisy = p.source.model.type == "noisy_composite"

    data_points[(problem, n, alg, depth, noisy)] = p.value


problems = ["maxcut", "partition", "vertex_cover"]
ns = [5, 6, 7, 8, 9, 10]
algs = ["qaoa", "wsqaoa", "wsinitqaoa", "rqaoa"]
noisys = [False, True]
depths = [1, 2, 3, 4]

problem_name = {
    "maxcut": "Max-Cut",
    "partition": "Partition",
    "vertex_cover": "Vertex Cover",
}
alg_name = {
    "qaoa": "QAOA",
    "wsqaoa": "WSQAOA",
    "wsinitqaoa": "WS-Init-QAOA",
    "rqaoa": "RQAOA",
}

headers = (
    "value",
    "problem",
    "model",
    "algorithm",
    "n",
)

rows = []

for problem in problems:
    for n in ns:
        approxmiation_values = [
            p.value
            for p in apply_filter(
                params,
                f"and and graphs.problem {problem} source.type approximation graphs.size {n}",
            )
        ]
        approxmiation_average = sum(approxmiation_values) / len(approxmiation_values)
        random_values = [
            p.value
            for p in apply_filter(
                params,
                f"and and graphs.problem {problem} source.type random graphs.size {n}",
            )
        ]
        random_average = sum(random_values) / len(random_values)

        rows.append((random_average, problem_name[problem], "Random", "", n))
        rows.append((approxmiation_average, problem_name[problem], "Classical", "", n))

    measurements = {(alg, noisy): [None] * len(ns) for alg in algs for noisy in noisys}
    for alg in algs:
        for noisy in noisys:
            for n in ns:
                ys = []
                for depth in depths:
                    key = (problem, n, alg, depth, noisy)
                    if key not in data_points:
                        continue
                    ys.append(data_points[key])
                if len(ys) == 0:
                    continue
                y = sum(ys) / len(ys)
                measurements[(alg, noisy)][n - 5] = y

    for alg in algs:
        for noisy in noisys:
            for n in ns:
                value = measurements[(alg, noisy)][n - 5]
                model_name = "Noisy" if noisy else "Ideal"
                rows.append(
                    (value, problem_name[problem], model_name, alg_name[alg], n)
                )

with open("csvs/algorithm_comparison_n_qubits.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(rows)
