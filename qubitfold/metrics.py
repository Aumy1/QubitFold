"""Hardware-independent gate count and greedy parallel-depth estimates."""
from collections import Counter
from .core import Circuit


def metrics(circuit: Circuit):
    available = [0] * circuit.n_qubits
    for gate in circuit.gates:
        level = 1 + max(available[w] for w in gate.wires)
        for w in gate.wires:
            available[w] = level
    return {
        "gates": len(circuit.gates),
        "depth": max(available, default=0),
        "two_qubit_gates": sum(len(g.wires) == 2 for g in circuit.gates),
        "gate_types": dict(sorted(Counter(g.name for g in circuit.gates).items())),
    }
