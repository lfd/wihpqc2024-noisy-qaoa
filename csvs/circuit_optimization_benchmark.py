import time
import csv
import os
import networkx as nx
from qat.opt import NumberPartitioning, MaxCut, VertexCover
from create_optimized_qaoa_circuit import create_optimized_qaoa_circuit


def benchmark_graph(graph, n_layers=1, n=100):
    problem = VertexCover(graph)

    start = time.process_time()
    for _ in range(n):
        job = create_optimized_qaoa_circuit(problem, n_layers)
    elapsed = (time.process_time() - start) / n
    return elapsed


if __name__ == "__main__":
    out_file = "csvs/circuit_optimization_benchmark.csv"
    if not os.path.isfile(out_file):
        n_lines = 0
    else:
        with open(out_file) as file:
            n_lines = len([l for l in file.readlines() if len(l.strip()) > 0])

    counter = 0
    with open(out_file, "a") as file:
        writer = csv.writer(file)
        if counter < n_lines:
            counter += 1
        else:
            writer.writerow(("problem", "size", "n_qaoa_layers", "seed", "runtime"))

        for n in [5, 6, 7, 8, 9, 10]:
            for n_layers in range(1, 4 + 1):
                random_graphs = [
                    nx.generators.random_graphs.erdos_renyi_graph(n, 0.5, seed=seed)
                    for seed in range(100)
                ]
                complete_graph = nx.generators.complete_graph(n)

                for seed, graph in enumerate(random_graphs):
                    if counter < n_lines:
                        counter += 2
                        continue
                    benchmark = benchmark_graph(graph, n_layers)
                    writer.writerow(("maxcut", n, n_layers, seed, benchmark))
                    writer.writerow(("vertex_cover", n, n_layers, seed, benchmark))
                    file.flush()

                complete_benchmark = benchmark_graph(complete_graph, n_layers)
                for seed in range(100):
                    if counter < n_lines:
                        counter += 1
                        continue
                    writer.writerow(
                        ("partition", n, n_layers, seed, complete_benchmark)
                    )
                file.flush()
