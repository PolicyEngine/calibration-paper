"""The dependency-free demo runs every classical method end to end.

Runs on the base install (no torch, no populace), so it is CI's proof that the
registry, adapters, solvers, and diagnostics all connect.
"""

from __future__ import annotations

import numpy as np

from calibration_paper.methods import MethodFamily, list_methods
from calibration_paper.smoke import run_classical_demo


def test_demo_runs_every_classical_method() -> None:
    """One finite result per classical method, targets reproduced, ESS sensible."""
    results = run_classical_demo(seed=0, n_records=200)
    assert [r.key for r in results] == list_methods(MethodFamily.CLASSICAL)
    for result in results:
        assert np.isfinite(result.outcome.weights).all(), result.key
        assert np.isfinite(result.relative_errors).all(), result.key
        # On the feasible demo surface every classical method converges.
        assert result.outcome.converged, result.key
        # ESS is a positive count no larger than the record count.
        assert 0.0 < result.diagnostics.effective_sample_size <= 200.0 + 1e-6


def test_demo_positive_methods_produce_no_negative_weights() -> None:
    """The demo confirms the declared positivity guarantees on real weights."""
    from calibration_paper.methods import get_method

    for result in run_classical_demo(seed=1):
        if get_method(result.key).guarantees_positive:
            assert result.diagnostics.negative_weight_share == 0.0, result.key


def test_demo_is_deterministic() -> None:
    """Same seed, identical weights -- the classical methods are deterministic."""
    first = run_classical_demo(seed=2)
    second = run_classical_demo(seed=2)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.outcome.weights, b.outcome.weights)
