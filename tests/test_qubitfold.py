import math
import numpy as np
import pytest
from qubitfold import (
    Circuit, Gate, optimize, simulate, unitary,
    equivalent_up_to_global_phase, metrics,
)
from qubitfold.simulate import gate_matrix


def c(n, *gates):
    return Circuit(n, gates)


@pytest.mark.parametrize("name", ["H", "X", "Y", "Z"])
def test_single_involutions(name):
    original = c(1, Gate(name, (0,)), Gate(name, (0,)))
    optimized = optimize(original)
    assert optimized.gates == ()
    assert equivalent_up_to_global_phase(original, optimized)


@pytest.mark.parametrize("name", ["CNOT", "CZ", "SWAP"])
def test_two_qubit_involutions(name):
    original = c(2, Gate(name, (0, 1)), Gate(name, (0, 1)))
    optimized = optimize(original)
    assert optimized.gates == ()
    assert equivalent_up_to_global_phase(original, optimized)


def test_nonadjacent_disjoint_rewrites():
    original = c(3, Gate("H", (0,)), Gate("X", (2,)),
                 Gate("H", (0,)), Gate("X", (2,)))
    assert optimize(original).gates == ()


def test_rotation_merge_across_independent_gate():
    original = c(2, Gate("RZ", (0,), 0.2), Gate("H", (1,)),
                 Gate("RZ", (0,), 0.3))
    rewritten = optimize(original)
    assert len(rewritten.gates) == 2
    assert rewritten.gates[0].theta == pytest.approx(0.5)
    assert equivalent_up_to_global_phase(original, rewritten)


def test_rotation_cancel_and_global_phase():
    original = c(1, Gate("RX", (0,), math.pi), Gate("RX", (0,), math.pi))
    assert optimize(original).gates == ()
    assert equivalent_up_to_global_phase(original, optimize(original))


def test_overlap_prevents_illegal_rewrites():
    original = c(2, Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("H", (0,)))
    assert len(optimize(original).gates) == 3
    assert equivalent_up_to_global_phase(original, optimize(original))


def test_reverse_cnot_orientation_not_cancelled():
    original = c(2, Gate("CNOT", (0, 1)), Gate("CNOT", (1, 0)))
    assert len(optimize(original).gates) == 2


def test_bell_state():
    circuit = c(2, Gate("H", (0,)), Gate("CNOT", (0, 1)))
    actual = simulate(circuit)
    expected = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_cnot_wire_orientation():
    circuit = c(2, Gate("X", (1,)), Gate("CNOT", (1, 0)))
    actual = simulate(circuit)
    np.testing.assert_allclose(actual, [0, 0, 0, 1])


def test_all_gate_matrices_unitary():
    gates = [
        Gate(name, (0,)) for name in ["H", "X", "Y", "Z"]
    ] + [
        Gate(name, (0,), 0.783) for name in ["RX", "RY", "RZ"]
    ] + [
        Gate(name, (0, 1)) for name in ["CNOT", "CZ", "SWAP"]
    ]
    for gate in gates:
        matrix = gate_matrix(gate)
        np.testing.assert_allclose(
            matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-12
        )


def test_randomized_semantic_equivalence():
    rng = np.random.default_rng(2027)
    for n in (2, 3, 4):
        for _ in range(25):
            gates = []
            for _ in range(30):
                kind = str(rng.choice(["H", "X", "Y", "Z", "RX",
                                       "RY", "RZ", "CNOT", "CZ", "SWAP"]))
                k = 2 if kind in ("CNOT", "CZ", "SWAP") else 1
                wires = tuple(int(x) for x in rng.choice(n, k, replace=False))
                theta = float(rng.normal()) if kind in ("RX", "RY", "RZ") else None
                gates.append(Gate(kind, wires, theta))
            original = Circuit(n, tuple(gates))
            assert equivalent_up_to_global_phase(original, optimize(original))


def test_optimizer_idempotence():
    circuit = c(2, Gate("H", (0,)), Gate("RX", (1,), 1.0),
                Gate("H", (0,)), Gate("RX", (1,), 2.0))
    assert optimize(optimize(circuit)) == optimize(circuit)


def test_depth_parallelism():
    circuit = c(3, Gate("H", (0,)), Gate("H", (2,)),
                Gate("CNOT", (0, 1)), Gate("X", (2,)))
    assert metrics(circuit)["gates"] == 4
    assert metrics(circuit)["depth"] == 2
    assert metrics(circuit)["two_qubit_gates"] == 1


@pytest.mark.parametrize("bad", [
    lambda: Gate("FOO", (0,)),
    lambda: Gate("CNOT", (0, 0)),
    lambda: Gate("RX", (0,)),
    lambda: Gate("H", (-1,)),
    lambda: Circuit(1, (Gate("X", (1,)),)),
])
def test_invalid_circuits(bad):
    with pytest.raises((ValueError, TypeError)):
        bad()


def test_json_round_trip():
    original = c(2, Gate("H", (0,)), Gate("RZ", (1,), 0.4))
    assert Circuit.from_dict(original.as_dict()) == original


def test_non_equivalent_circuits_detected():
    a, b = c(1, Gate("X", (0,))), c(1, Gate("Z", (0,)))
    assert not equivalent_up_to_global_phase(a, b)


def test_unitary_rejects_large_matrices():
    with pytest.raises(ValueError):
        unitary(c(7))
