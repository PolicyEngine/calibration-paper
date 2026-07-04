"""A dependency-free end-to-end calibration run for the CI smoke path.

Runs every *classical* method (the pure-NumPy ones, always available) on a small
feasible synthetic problem and returns their diagnostics, so ``cal demo`` and the
CI tests push real methods through the real diagnostics without torch or
populace. The gradient methods are exercised by ``tests/test_gradient.py`` when
the ``methods`` extra is installed.
"""

from __future__ import annotations

from calibration_paper import methods as method_registry
from calibration_paper.methods import MethodFamily, MethodResult, run_method
from calibration_paper.synthetic import feasible_problem


def run_classical_demo(*, seed: int = 0, n_records: int = 200) -> list[MethodResult]:
    """Calibrate a feasible synthetic problem with every classical method.

    Args:
        seed: Seed for the synthetic problem and the (deterministic) methods.
        n_records: Records in the synthetic problem.

    Returns:
        One :class:`~calibration_paper.methods.MethodResult` per classical method,
        in registry order.
    """
    problem = feasible_problem(n_records=n_records, n_targets=8, seed=seed).problem
    results: list[MethodResult] = []
    for key in method_registry.list_methods(MethodFamily.CLASSICAL):
        results.append(run_method(key, problem, seed=seed))
    return results
