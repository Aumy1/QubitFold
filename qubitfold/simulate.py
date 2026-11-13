"""Small statevector backend used to independently verify rewrite passes."""
import math
import numpy as np
from .core import Circuit, Gate

SQRT2 = math.sqrt(2)
MATRICES = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / SQRT2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
    "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                      [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
}


def gate_matrix(gate: Gate):
    if gate.name in MATRICES:
        return MATRICES[gate.name]
    t = gate.theta
    c = math.cos(t / 2)
    s = math.sin(t / 2)
    if gate.name == "RX":
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    if gate.name == "RY":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if gate.name == "RZ":
        return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
    raise ValueError(f"Unknown gate {gate.name}")


def apply_gate(state: np.ndarray, gate: Gate, n: int):
    """Apply a k-qubit operation to arbitrarily ordered axes.

    Axis ordering follows gate.wires, so CNOT(control, target) is directional.
    """
    selected = list(gate.wires)
    rest = [w for w in range(n) if w not in selected]
    axes = selected + rest
    inverse = np.argsort(axes)
    transposed = state.reshape([2] * n).transpose(axes)
    cols = transposed.reshape(2 ** len(selected), -1)
    result = (gate_matrix(gate) @ cols).reshape([2] * n)
    return result.transpose(inverse).reshape(-1)


def simulate(circuit: Circuit, initial: np.ndarray | None = None):
    dim = 1 << circuit.n_qubits
    if initial is None:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1
    else:
        state = np.asarray(initial, dtype=complex).copy()
        if state.shape != (dim,):
            raise ValueError(f"Expected state of shape ({dim},)")
        if not np.isclose(np.linalg.norm(state), 1):
            raise ValueError("Initial state must be normalized")
    for gate in circuit.gates:
        state = apply_gate(state, gate, circuit.n_qubits)
    return state


def unitary(circuit: Circuit, max_qubits=6):
    """Full circuit matrix. Limited to small sizes for memory/time safety."""
    if circuit.n_qubits > max_qubits:
        raise ValueError(f"Full-matrix verification limited to {max_qubits} qubits")
    dim = 1 << circuit.n_qubits
    cols = [simulate(circuit, np.eye(dim, dtype=complex)[:, i])
            for i in range(dim)]
    return np.column_stack(cols)


def equivalent_up_to_global_phase(a: Circuit, b: Circuit, atol=1e-9):
    """Compare *full unitaries*, not only output for |0...0>."""
    if a.n_qubits != b.n_qubits:
        return False
    x, y = unitary(a), unitary(b)
    idx = np.unravel_index(np.abs(y).argmax(), y.shape)
    if abs(y[idx]) < atol:
        return False
    factor = x[idx] / y[idx]
    return np.isclose(abs(factor), 1, atol=atol) and np.allclose(
        x, factor * y, atol=atol, rtol=0
    )
