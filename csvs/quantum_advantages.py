import csv


data = {}

rows = []
with open("csvs/algorithm_comparison_n_layers.csv") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        rows.append((row, "By layers"))
with open("csvs/algorithm_comparison_n_qubits.csv") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        rows.append((row, "By qubits"))

problems = set()
algorithms = set()
ns = {}


for (performance, problem, model, algorithm, n), by in rows:
    performance = float(performance)
    n = int(n)
    key = by, problem, algorithm, n, model
    data[key] = performance
    problems.add(problem)

    if algorithm != "":
        algorithms.add(algorithm)

    if by not in ns:
        ns[by] = set()

    ns[by].add(n)

with open("csvs/quantum_advantages.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(("type", "problem", "algorithm", "advantage"))
    for problem in problems:
        for algorithm in algorithms:
            for by in ns:
                is_advantage = False
                for n in ns[by]:
                    key_ideal = by, problem, algorithm, n, "Ideal"
                    key_noisy = by, problem, algorithm, n, "Noisy"
                    key_approx = by, problem, "", n, "Classical Approximation"

                    is_advantage = is_advantage or (
                        data[key_ideal] > data[key_approx]
                        or data[key_noisy] > data[key_approx]
                    )
                if is_advantage:
                    value = 1
                else:
                    value = 0

                writer.writerow((by, problem, algorithm, value))
