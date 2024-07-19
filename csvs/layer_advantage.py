import json
from parse_params import Params
import csv

files = [
    "results/main_evaluation.txt",
]

params = []

for file_name in files:
    with open(file_name) as file:
        lines = file.readlines()
    params.extend([Params(**json.loads(line)) for line in lines])

data = {}
for param in params:
    problem = param.graphs.problem
    algorithm = param.source.algorithm.type

    noise = param.source.model.noise
    time = param.source.model.time

    if noise is None:
        noise = 0.0
    if time is None:
        time = 0.0

    size = param.graphs.size
    layer = param.source.algorithm.n_layers

    key = problem, algorithm, layer, size, noise, time
    data[key] = param.raw

noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
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

rows = []
for problem in ["maxcut", "partition", "vertex_cover"]:
    for algorithm in ["qaoa", "wsqaoa", "wsinitqaoa", "rqaoa"]:
        for layer in [2, 3]:
            for noise in noise_levels:
                for time in noise_levels:
                    values_p = []
                    values_p_minus_1 = []
                    for size in range(5, 11):
                        values_p.extend(
                            data[(problem, algorithm, layer, size, noise, time)]
                        )
                        values_p_minus_1.extend(
                            data[(problem, algorithm, layer - 1, size, noise, time)]
                        )
                    layer_advantage = [
                        a / b for a, b in zip(values_p, values_p_minus_1)
                    ]
                    avg = sum(layer_advantage) / len(layer_advantage)
                    rows.append(
                        (
                            problem_name[problem],
                            alg_name[algorithm],
                            layer,
                            noise,
                            time,
                            avg,
                        )
                    )

headers = (
    "problem",
    "algorithm",
    "layer",
    "depolarizing_noise",
    "thermal_relaxation_noise",
    "relative_performance",
)

with open("csvs/layer_advantage.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(rows)
