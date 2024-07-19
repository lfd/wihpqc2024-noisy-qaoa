.ONESHELL:

N_THREADS?=1

PYTHON := PYTHONPATH=$(PYTHONPATH):. .venv/bin/python
PIP := .venv/bin/pip
ACTIVATE := .venv/bin/activate

install:
	$(PIP) install -r requirements.txt

RESULTS_DIR := results
RESULTS_SCRIPTS := $(wildcard $(RESULTS_DIR)/*.bash)
RESULTS := $(patsubst $(RESULTS_DIR)/%.bash, $(RESULTS_DIR)/%.txt, $(RESULTS_SCRIPTS))

$(RESULTS_DIR)/%.txt: $(RESULTS_DIR)/%.bash
	source $(ACTIVATE)
	bash $< $(N_THREADS)

results: $(RESULTS)

csvs/algorithm_comparison_n_layers.csv: csvs/algorithm_comparison_n_layers.py results/main_evaluation.txt results/bounds.txt
	$(PYTHON) $<

csvs/algorithm_comparison_n_qubits.csv: csvs/algorithm_comparison_n_qubits.py results/main_evaluation.txt results/bounds.txt
	$(PYTHON) $<

csvs/layer_advantage.csv: csvs/layer_advantage.py results/main_evaluation.txt
	$(PYTHON) $<

csvs/classical_approximation_benchmark.csv: csvs/classical_approximation_benchmark.py
	$(PYTHON) $<

csvs/circuit_optimization_benchmark.csv: csvs/circuit_optimization_benchmark.py
	$(PYTHON) $<

csvs/performance_by_runtime.csv: csvs/performance_by_runtime.py csvs/circuit_optimization_benchmark.csv csvs/classical_approximation_benchmark.csv results/main_evaluation.txt
	$(PYTHON) $<

csvs: csvs/algorithm_comparison_n_layers.csv  csvs/algorithm_comparison_n_qubits.csv  csvs/circuit_optimization_benchmark.csv  csvs/classical_approximation_benchmark.csv  csvs/layer_advantage.csv  csvs/performance_by_runtime.csv

plots:
	docker build -t qsw-noisy-qaoa . && docker run -v $(pwd)/img-pdf:/app/img-pdf -v $(pwd)/img-tikz:/app/img-tikz qsw-noisy-qaoa

.PHONY: install results csvs plots