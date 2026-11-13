"""Conservative, semantics-preserving peephole compiler passes.

Rules commute only gates on disjoint wires and simplify same-wire operations.
Rotation changes by 2*pi are ignored as they alter only global phase.
"""
import math
from .core import Circuit, Gate, ROTATIONS

SELF_INVERSE = {"H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"}
TOL = 1e-12


def _canonical_angle(theta: float):
    return math.remainder(theta, 2 * math.pi)


def optimize(circuit: Circuit):
    result: list[Gate] = []
    for gate in circuit.gates:
        if gate.name in ROTATIONS:
            angle = _canonical_angle(gate.theta)
            if abs(angle) < TOL:
                continue
            gate = Gate(gate.name, gate.wires, angle)

        # Find most recent gate whose wires overlap this gate.
        # Any intervening gates act on DISJOINT wires and commute.
        touched = set(gate.wires)
        previous = next(
            (i for i in range(len(result) - 1, -1, -1)
             if touched.intersection(result[i].wires)),
            None,
        )
        if previous is not None:
            candidate = result[previous]
            if candidate.name == gate.name and candidate.wires == gate.wires:
                if gate.name in SELF_INVERSE:
                    result.pop(previous)
                    continue
                if gate.name in ROTATIONS:
                    merged = _canonical_angle(candidate.theta + gate.theta)
                    if abs(merged) < TOL:
                        result.pop(previous)
                    else:
                        result[previous] = Gate(gate.name, gate.wires, merged)
                    continue
        result.append(gate)
    return Circuit(circuit.n_qubits, tuple(result))
