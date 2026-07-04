"""``cal demo``: run every classical method on a synthetic problem.

The dependency-free end-to-end demo: build a feasible synthetic calibration
problem, run every classical method (:mod:`calibration_paper.smoke`), and print
each one's convergence, worst target miss, effective sample size, and
negative-weight incidence. Proof that the registry, adapters, solvers, and
diagnostics all connect -- and a first look at how the classical methods differ
on one problem -- without torch, populace, or R. The gradient methods need the
``methods`` extra; the sweep (issue #2) runs the full surface on the frozen
inputs.
"""

from __future__ import annotations

import argparse

from calibration_paper.smoke import run_classical_demo


def main(argv: list[str] | None = None) -> int:
    """Run the classical demo and print a per-method diagnostics table.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]`` via argparse.

    Returns:
        ``0`` if every method converged on the feasible demo surface, else ``1``
        (so scripting can detect a regression that broke a solver).
    """
    parser = argparse.ArgumentParser(
        prog="cal demo",
        description="Run every classical method on a synthetic feasible problem.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the synthetic problem."
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=200,
        help="Number of records in the synthetic problem.",
    )
    args = parser.parse_args(argv)

    results = run_classical_demo(seed=args.seed, n_records=args.n_records)

    key_width = max(len(r.key) for r in results)
    header = f"{'method':<{key_width}}  conv   max_rel_err       ess  ess%   neg%"
    print(f"Classical calibration demo (seed={args.seed}, n={args.n_records})")
    print(header)
    print("-" * len(header))
    all_converged = True
    for result in results:
        diag = result.diagnostics
        max_abs_err = float(max(abs(e) for e in result.relative_errors))
        converged = result.outcome.converged
        all_converged = all_converged and converged
        print(
            f"{result.key:<{key_width}}  "
            f"{'yes' if converged else 'no ':<4}  "
            f"{max_abs_err:11.2e}  "
            f"{diag.effective_sample_size:8.1f}  "
            f"{diag.ess_ratio * 100:4.1f}  "
            f"{diag.negative_weight_share * 100:5.1f}"
        )
    return 0 if all_converged else 1


if __name__ == "__main__":  # pragma: no cover - module-run convenience
    raise SystemExit(main())
