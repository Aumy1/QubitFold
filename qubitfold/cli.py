"""CLI: qubitfold optimize file.json [-o output.json]."""
import argparse
import json
from pathlib import Path
from .core import Circuit
from .optimize import optimize
from .metrics import metrics
from .simulate import equivalent_up_to_global_phase


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="qubitfold", description="Verified quantum peephole optimization"
    )
    parser.add_argument("source", type=Path, help="Circuit JSON to optimize")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip full-unitary validation for large circuits")
    args = parser.parse_args(argv)
    original = Circuit.from_dict(json.loads(args.source.read_text()))
    optimized = optimize(original)
    report = {
        "original": metrics(original),
        "optimized": metrics(optimized),
    }
    if not args.skip_verify:
        report["equivalent_up_to_global_phase"] = equivalent_up_to_global_phase(
            original, optimized
        )
        if not report["equivalent_up_to_global_phase"]:
            raise RuntimeError("Optimizer changed the circuit unitary")
    if args.output is not None:
        args.output.write_text(json.dumps(optimized.as_dict(), indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
