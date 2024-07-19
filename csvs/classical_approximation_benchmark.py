import time
import networkx as nx
import numpy as np
import cvxpy as cp
import csv
import random
import os
from run_benchmarks import (
    perform_maxcut_sdp,
    apply_gw_rounding,
    partition_greedy,
    vertex_cover_approx,
)


def maxcut_approx(graph):
    return apply_gw_rounding(perform_maxcut_sdp(graph))


def gen_problem(problem_type, size, seed):
    if problem_type in ["maxcut", "vertex_cover"]:
        graph = nx.generators.random_graphs.erdos_renyi_graph(size, 0.5, seed=seed)
        return graph
    if problem_type == "partition":
        np.random.seed(seed)
        numbers = np.random.rand(size)
        return numbers


approximation_algs = {
    "maxcut": maxcut_approx,
    "partition": partition_greedy,
    "vertex_cover": vertex_cover_approx,
}

out_file_name = "csvs/classical_approximation_benchmark.csv"

if not os.path.isfile(out_file_name):
    n_lines = 0
else:
    with open(out_file_name) as file:
        n_lines = len([l for l in file.readlines() if len(l.strip()) > 0])

counter = 0
with open(out_file_name, "a") as file:
    writer = csv.writer(file)

    def writerow(row):
        global counter
        if counter < n_lines:
            counter += 1
        else:
            writer.writerow(row)

    writerow(("problem", "size", "seed", "runtime"))
    n = 1000
    for problem_type in ["maxcut", "partition", "vertex_cover"]:
        approx = approximation_algs[problem_type]
        for size in range(5, 10 + 1):
            for seed in range(100):
                problem = gen_problem(problem_type, size, seed)
                elapsed = 0
                if counter >= n_lines:
                    start = time.process_time()
                    for _ in range(n):
                        approx(problem)
                    elapsed = (time.process_time() - start) / n
                writerow((problem_type, size, seed, elapsed))
                file.flush()
