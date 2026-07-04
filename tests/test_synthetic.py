"""Properties of the synthetic problem builders.

The builders back the CI smoke path and the scale curves; these tests pin the
properties the method tests rely on (feasibility, positivity of ``w0``, the
contradiction in the infeasible surface).
"""

from __future__ import annotations

import numpy as np

from calibration_paper.synthetic import (
    feasible_problem,
    infeasible_problem,
    margins_problem,
)


def test_feasible_problem_targets_are_reproduced_by_true_weights() -> None:
    """``b == A @ true_weights`` exactly, so the surface is feasible by design."""
    sp = feasible_problem(n_records=300, n_targets=10, seed=0)
    assert sp.true_weights is not None
    np.testing.assert_allclose(
        sp.problem.estimates(sp.true_weights), sp.problem.target, rtol=1e-12
    )
    assert (sp.problem.initial_weights > 0.0).all()


def test_feasible_problem_scales_with_the_knobs() -> None:
    """The scale knobs set the matrix shape (for the 10^2 -> 10^4 curves)."""
    sp = feasible_problem(n_records=1000, n_targets=100, seed=1)
    assert sp.problem.n_records == 1000
    assert sp.problem.n_targets == 100


def test_feasible_problem_has_two_target_families() -> None:
    """Targets are split across two families so held-out splits have structure."""
    sp = feasible_problem(n_records=100, n_targets=8, seed=2)
    assert set(sp.problem.target_families) == {"family_a", "family_b"}


def test_margins_problem_targets_are_feasible_counts() -> None:
    """Row/column margins come from a common table, so they are jointly feasible."""
    sp = margins_problem(n_rows=4, n_cols=5, per_cell=20, seed=3)
    assert sp.true_weights is not None
    np.testing.assert_allclose(
        sp.problem.estimates(sp.true_weights), sp.problem.target, rtol=1e-12
    )
    # The constraint rows are 0/1 indicators (membership).
    dense = sp.problem.dense()
    assert set(np.unique(dense)).issubset({0.0, 1.0})


def test_infeasible_problem_has_no_true_weights_and_contradicts_itself() -> None:
    """The two component rows are identical functionals with different targets."""
    sp = infeasible_problem(n_records=150, seed=4)
    assert sp.true_weights is None
    dense = sp.problem.dense()
    # Rows 1 and 2 are the same functional...
    np.testing.assert_array_equal(dense[1], dense[2])
    # ... asked to equal different values.
    assert sp.problem.target[1] != sp.problem.target[2]
