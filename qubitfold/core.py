"""Immutable circuit IR with explicit wire order (wire 0 is most significant)."""
from dataclasses import dataclass
import math

SINGLE = {"H", "X", "Y", "Z", "RX", "RY", "RZ"}
DOUBLE = {"CNOT", "CZ", "SWAP"}
ROTATIONS = {"RX", "RY", "RZ"}
ALL = SINGLE | DOUBLE


@dataclass(frozen=True)
class Gate:
    name: str
    wires: tuple[int, ...]
    theta: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", self.name.upper())
        object.__setattr__(self, "wires", tuple(self.wires))
        if self.name not in ALL:
            raise ValueError(f"Unsupported gate: {self.name}")
        arity = 1 if self.name in SINGLE else 2
        if len(self.wires) != arity or len(set(self.wires)) != arity:
            raise ValueError(f"{self.name} requires {arity} distinct wires")
        if any(type(w) is not int or w < 0 for w in self.wires):
            raise ValueError("Wire indices must be non-negative integers")
        if self.name in ROTATIONS:
            if self.theta is None or not math.isfinite(float(self.theta)):
                raise ValueError("Rotation requires finite theta in radians")
            object.__setattr__(self, "theta", float(self.theta))
        elif self.theta is not None:
            raise ValueError(f"{self.name} does not accept theta")

    def as_dict(self):
        data = {"name": self.name, "wires": list(self.wires)}
        if self.theta is not None:
            data["theta"] = self.theta
        return data


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    gates: tuple[Gate, ...]

    def __post_init__(self):
        object.__setattr__(self, "gates", tuple(self.gates))
        validate_circuit(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "n_qubits" not in data or "gates" not in data:
            raise ValueError("Expected {n_qubits: int, gates: list}")
        gates = tuple(
            Gate(g["name"], tuple(g["wires"]), g.get("theta"))
            for g in data["gates"]
        )
        return cls(data["n_qubits"], gates)

    def as_dict(self):
        return {"n_qubits": self.n_qubits, "gates": [g.as_dict() for g in self.gates]}


def validate_circuit(circuit: Circuit):
    if type(circuit.n_qubits) is not int or circuit.n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    for gate in circuit.gates:
        if not isinstance(gate, Gate):
            raise TypeError("Circuit gates must be Gate objects")
        if max(gate.wires) >= circuit.n_qubits:
            raise ValueError(f"{gate.name} references an out-of-range wire")
