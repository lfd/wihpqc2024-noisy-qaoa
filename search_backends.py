import os
import json
from collections import defaultdict
import argparse
from typing import Any


backends_root = "qiskit-terra/qiskit/providers/fake_provider/backends"


class Backend:
    def __init__(self, name: str, conf: dict, props: dict):
        self.name = name
        self.conf = conf
        self.props = props
        self.n_qubits = conf["n_qubits"]
        self.couplings = conf["coupling_map"]
        self.n_couplings = None if self.couplings is None else len(self.couplings)
        if self.n_couplings is None or self.n_qubits < 10:
            self.coupling_density = None
        else:
            self.coupling_density = len(self.couplings) / (
                (self.n_qubits - 1) * self.n_qubits
            )

        self.values_T1 = get_qubit_values(props, "T1")
        self.values_T2 = get_qubit_values(props, "T2")
        self.values_T1_over_T2 = [
            t1 / t2 for t1, t2 in zip(self.values_T1, self.values_T2)
        ]
        self.prep_0_meas_1 = get_qubit_values(props, "prob_meas1_prep0")
        self.prep_1_meas_0 = get_qubit_values(props, "prob_meas0_prep1")
        self.readout_error = get_qubit_values(props, "readout_error")
        self.gate_names = get_gate_names(props)
        self.gates = {}
        for g in self.gate_names:
            gate_props = get_gate_properties(props, g)
            self.gates[g] = gate_props


def average(l: list[float]) -> float:
    l = [v for v in l if v is not None]
    if len(l) == 0:
        return None
    return sum(l) / len(l)


def median(l: list[float]) -> float:
    l = [v for v in l if v is not None]
    if len(l) == 0:
        return None
    l.sort()
    if len(l) % 2 == 0:
        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2
    else:
        return l[(len(l) - 1) // 2]


def largest(l: list[float]) -> float:
    l = [v for v in l if v is not None]
    if len(l) == 0:
        return None
    return max(l)


def smallest(l: list[float]) -> float:
    l = [v for v in l if v is not None]
    if len(l) == 0:
        return None
    return min(l)


def get_qubit_values(props: dict, name: str) -> list[Any]:
    qubits = props["qubits"]
    values = []
    for qubit in qubits:
        for spec in qubit:
            if spec["name"] == name:
                values.append(spec["value"])
    return values


def compute_qubit_average(props: dict, name: str) -> float:
    return average(get_qubit_values(props, name))


def compute_qubit_median(props: dict, name: str) -> float:
    return median(get_qubit_values(props, name))


def get_gate_names(props: dict) -> set[str]:
    gate_names = set()
    for gate in props["gates"]:
        gate_names.add(gate["gate"])
    return gate_names


def get_gate_properties(props: dict, gate_name: str) -> tuple[float, float, float]:
    lengths = []
    errors = []

    for gate in props["gates"]:
        if gate["gate"] != gate_name:
            continue
        n_qubits = len(gate["qubits"])
        for param in gate["parameters"]:
            if param["name"] == "gate_error":
                errors.append(param["value"])
            if param["name"] == "gate_length":
                lengths.append(param["value"])

    avg_length = median(lengths)
    avg_error = median(errors)
    return n_qubits, avg_length, avg_error


accs = {
    "average": average,
    "median": median,
    "max": largest,
    "min": smallest,
}

acc_names = sorted(list(accs.keys()))


def main(
    acc_name: str,
    inner_backend_acc_name: str,
    backend_name: str,
    gate_filter: str,
    verbose: bool,
):
    accumulate = accs[acc_name]
    inner_backend_accumulate = accs[inner_backend_acc_name]
    backend_dirs = [f for f in os.scandir(backends_root) if f.is_dir()]
    backends = []

    for d in backend_dirs:
        if backend_name is not None and d.name != backend_name:
            continue
        conf_path = os.path.join(d.path, f"conf_{d.name}.json")
        props_path = os.path.join(d.path, f"props_{d.name}.json")

        if not os.path.exists(conf_path) or not os.path.exists(props_path):
            continue

        with open(conf_path) as file:
            conf = json.load(file)
        with open(props_path) as file:
            props = json.load(file)

        backends.append(Backend(d.name, conf, props))

    t1 = []
    t2 = []
    readout = []
    prep0_meas1 = []
    prep1_meas0 = []
    t1_over_t2 = []
    n_qubits = []
    gates = defaultdict(lambda: (list(), list(), list(), list(), list()))
    connectivity = []
    coupling_densities = []

    for backend in backends:
        if backend.n_couplings is None:
            connectivity.append(None)
        else:
            connectivity.append(
                backend.n_couplings / (backend.n_qubits * (backend.n_qubits - 1))
            )
        coupling_densities.append(backend.coupling_density)

        if gate_filter is not None and gate_filter not in backend.gate_names:
            continue
        backend_t1 = inner_backend_accumulate(backend.values_T1) * 1000
        backend_t2 = inner_backend_accumulate(backend.values_T2) * 1000
        t1.append(backend_t1)
        t2.append(backend_t2)
        t1_over_t2.append(inner_backend_accumulate(backend.values_T1_over_T2))
        prep0_meas1.append(inner_backend_accumulate(backend.prep_0_meas_1))
        prep1_meas0.append(inner_backend_accumulate(backend.prep_1_meas_0))
        readout.append(inner_backend_accumulate(backend.readout_error))
        gate_set = set()
        for name, (_, length, error) in backend.gates.items():
            gate_set.add(name)
            lengths, backends, errors, t1_portion, t2_portion = gates[name]
            lengths.append(length)
            errors.append(error)
            backends.append(backend.name)
            t1_portion.append(length / backend_t1)
            t2_portion.append(length / backend_t2)
        n_qubits.append(backend.n_qubits)

    acc_t1 = accumulate(t1)
    acc_t2 = accumulate(t2)
    acc_prep0_meas1 = accumulate(prep0_meas1)
    acc_prep1_meas0 = accumulate(prep1_meas0)
    acc_readout = accumulate(readout)
    n_qubits = accumulate(n_qubits)
    connectivity = accumulate(connectivity)
    coupling_density = accumulate(coupling_densities)
    print(
        f"n qubits: {n_qubits}, connectivity: {connectivity}, coupling density: {coupling_density}, t1: {accumulate(t1)}, t2: {accumulate(t2)}, t1 / t2: {accumulate(t1_over_t2)}, readout {acc_readout}, prep0 meas1: {acc_prep0_meas1}, prep1 meas0: {acc_prep1_meas0}"
    )
    for name, (lengths, backends, errors, t1_portion, t2_portion) in gates.items():
        avg_length = accumulate(lengths)
        avg_error = accumulate(errors)
        avg_t1_portion = accumulate(t1_portion)
        avg_t2_portion = accumulate(t2_portion)
        if verbose:
            print(
                f"{name} ({len(backends)} backends)\n"
                + f"length: {avg_length}, infidelity: {avg_error}\n"
                + f"t/T1: {avg_t1_portion}, t/avgT1: {avg_length / acc_t1}\n"
                + f"t/T2: {avg_t2_portion}, t/avgT2: {avg_length / acc_t2}\n"
            )
        else:
            print(f"{name} ({len(backends)}): {avg_length} {avg_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acc",
        choices=acc_names,
        required=True,
        help="Function to combine backend results to single value",
    )
    parser.add_argument(
        "--inner_backend_acc",
        choices=acc_names,
        help="Function to combine qubit results of one backend to single value (the same as --acc by default)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="If set the name of a backend (e.g. almaden), only show results for this backend",
    )
    parser.add_argument(
        "--gate_filter",
        default=None,
        help='If set (e.g. to "u3"), only show use results for backends supporting this gate',
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="If set, show more results (e.g. ratio of T1 and gate duration)",
    )
    args = parser.parse_args()

    inner_backend_acc = args.inner_backend_acc
    if inner_backend_acc is None:
        inner_backend_acc = args.acc

    main(args.acc, inner_backend_acc, args.backend, args.gate_filter, args.v)
