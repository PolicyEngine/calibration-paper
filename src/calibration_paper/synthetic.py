"""Seeded synthetic calibration problems with known solutions.

Nothing here imports torch or populace: pure numpy/scipy, so ``cal demo`` and
the CI tests run on the base install and every calibrator's *contract* is
checkable against a problem whose answer is known.

Three builders, one per thing a calibrator test needs:

* :func:`feasible_problem` -- a random design matrix whose target vector is
  ``A @ w_true`` for a known positive ``w_true`` inside sensible bounds, so every
  method (classical and gradient) *can* reproduce the targets exactly, and the
  test can assert ``A @ w == b`` (to tolerance) and, where the method guarantees
  it, positivity and bound-respect.
* :func:`margins_problem` -- a raking-style two-way-margins surface (indicator
  rows summing counts), the classical margins-only regime where raking/IPF is
  the textbook method. Feasible by construction.
* :func:`infeasible_problem` -- a surface with contradictory targets (the same
  aggregate asked to equal two different values), so no reweighting can satisfy
  it: the failure-mode comparison (hard-constraint infeasibility vs soft-loss
  compromise) the paper runs.

The scale knob (``n_records``, ``n_targets``) lets the sweep's scale curves
(target counts 10^2 -> 10^4) run on synthetic problems before the frozen
populace surface is wired (issue #2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration_paper.problem import CalibrationProblem

__all__ = [
    "SyntheticProblem",
    "feasible_problem",
    "margins_problem",
    "infeasible_problem",
]


@dataclass(frozen=True)
class SyntheticProblem:
    """A synthetic calibration problem with its known true weights (if any).

    Attributes:
        problem: The :class:`CalibrationProblem` (``A``, ``b``, ``w0``).
        true_weights: The weight vector that reproduces ``b`` exactly, when the
            problem was built to be feasible; ``None`` for an infeasible surface.
    """

    problem: CalibrationProblem
    true_weights: np.ndarray | None


def feasible_problem(
    *,
    n_records: int = 200,
    n_targets: int = 8,
    seed: int = 0,
    ratio_low: float = 0.5,
    ratio_high: float = 2.0,
) -> SyntheticProblem:
    """A dense problem whose targets are exactly reproducible by a known ``w``.

    Draws a design matrix ``A`` and positive design weights ``w0``, then a known
    g-weight vector ``g_true`` uniform in ``[ratio_low, ratio_high]`` and sets
    ``b = A @ (w0 * g_true)``. So ``w_true = w0 * g_true`` reproduces ``b``
    exactly, and because the true ratios lie inside common bounds, every
    method's guarantees can be asserted: exact target reproduction for the
    smooth methods, and bound-respect for the bounded ones (whose bounds contain
    ``[ratio_low, ratio_high]``).

    Args:
        n_records: Number of records (matrix columns).
        n_targets: Number of targets (matrix rows). Kept well below
            ``n_records`` so the system is under-determined (the usual
            calibration regime) and a positive solution exists.
        seed: Seed for the draw.
        ratio_low: Lower bound of the true g-weights.
        ratio_high: Upper bound of the true g-weights.

    Returns:
        A :class:`SyntheticProblem` with ``true_weights`` set.
    """
    rng = np.random.default_rng(seed)
    # A mix of a constant row (a total/count target), signed continuous rows
    # (sum targets), and a couple of 0/1 indicator rows (subpopulation counts).
    matrix = rng.normal(size=(n_targets, n_records))
    matrix[0, :] = 1.0  # a population-total target
    if n_targets >= 3:
        matrix[1, :] = rng.lognormal(
            mean=0.0, sigma=0.5, size=n_records
        )  # positive sum
        matrix[2, :] = (rng.random(n_records) < 0.4).astype(np.float64)  # a count
    w0 = rng.lognormal(mean=6.0, sigma=0.3, size=n_records)
    g_true = rng.uniform(ratio_low, ratio_high, size=n_records)
    w_true = w0 * g_true
    target = matrix @ w_true
    problem = CalibrationProblem(
        matrix=matrix,
        target=target,
        initial_weights=w0,
        target_names=tuple(f"t{i}" for i in range(n_targets)),
        target_families=_round_robin_families(n_targets),
    )
    return SyntheticProblem(problem=problem, true_weights=w_true)


def margins_problem(
    *, n_rows: int = 3, n_cols: int = 4, per_cell: int = 25, seed: int = 0
) -> SyntheticProblem:
    """A two-way-margins raking surface: row and column count targets.

    Builds an ``n_rows x n_cols`` contingency of records (``per_cell`` records
    per cell, each weight 1 initially) and asks calibration to hit perturbed row
    and column margin *counts* -- the canonical raking / IPF problem. The
    targets are the margins of a known positive cell-weight table, so they are
    feasible and raking reproduces them exactly. The constraint rows are 0/1
    membership indicators, so ``A @ w`` is literally the vector of margin counts.

    Args:
        n_rows: Number of row categories.
        n_cols: Number of column categories.
        per_cell: Records per (row, col) cell.
        seed: Seed for the target perturbation.

    Returns:
        A :class:`SyntheticProblem` with feasible row/column margin targets.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_rows * n_cols
    n_records = n_cells * per_cell
    row_of = np.repeat(np.arange(n_rows), n_cols * per_cell)
    col_of = np.tile(np.repeat(np.arange(n_cols), per_cell), n_rows)

    # Row-margin and column-margin indicator rows (drop no category: the system
    # is consistent because the margins come from a common table).
    rows = []
    for r in range(n_rows):
        rows.append((row_of == r).astype(np.float64))
    for c in range(n_cols):
        rows.append((col_of == c).astype(np.float64))
    matrix = np.vstack(rows)

    w0 = np.ones(n_records, dtype=np.float64)
    # A known positive cell scaling, so the perturbed margins are feasible.
    cell_scale = rng.uniform(0.5, 1.8, size=n_records)
    w_true = w0 * cell_scale
    target = matrix @ w_true
    names = tuple(f"row{r}" for r in range(n_rows)) + tuple(
        f"col{c}" for c in range(n_cols)
    )
    families = ("rows",) * n_rows + ("cols",) * n_cols
    problem = CalibrationProblem(
        matrix=matrix,
        target=target,
        initial_weights=w0,
        target_names=names,
        target_families=families,
    )
    return SyntheticProblem(problem=problem, true_weights=w_true)


def infeasible_problem(*, n_records: int = 200, seed: int = 0) -> SyntheticProblem:
    """A surface with contradictory targets -- no reweighting can satisfy it.

    Two of the target rows are the *same* linear functional of the weights
    (identical matrix rows) but demand *different* values, so ``A @ w = b`` has
    no solution: a hard-constraint method cannot converge, and a soft-loss method
    settles on a compromise between the two. This is the deliberately infeasible
    surface for the failure-mode comparison (PLAN.md): hard-constraint
    infeasibility vs soft-loss compromise.

    Args:
        n_records: Number of records.
        seed: Seed for the draw.

    Returns:
        A :class:`SyntheticProblem` with ``true_weights=None`` (none exists).
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=0.0, sigma=0.5, size=n_records)
    w0 = rng.lognormal(mean=6.0, sigma=0.3, size=n_records)
    baseline_total = float(base @ w0)
    # Three rows: a feasible total, then the SAME functional asked to be both
    # 20% above and 20% below the baseline -- jointly impossible.
    matrix = np.vstack([np.ones(n_records), base, base])
    target = np.array(
        [float(w0.sum()), 1.2 * baseline_total, 0.8 * baseline_total],
        dtype=np.float64,
    )
    problem = CalibrationProblem(
        matrix=matrix,
        target=target,
        initial_weights=w0,
        target_names=("total", "component_high", "component_low"),
        target_families=("total", "contradiction", "contradiction"),
    )
    return SyntheticProblem(problem=problem, true_weights=None)


def _round_robin_families(n_targets: int) -> tuple[str, ...]:
    """Assign targets to two families in round-robin, for held-out splits.

    A synthetic stand-in for the paper's national/state target families: enough
    structure that :meth:`CalibrationProblem.subset_targets` has something to
    hold out, without pretending to be the real Ledger surface (issue #2).
    """
    families = ("family_a", "family_b")
    return tuple(families[i % 2] for i in range(n_targets))
