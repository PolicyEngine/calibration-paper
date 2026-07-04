"""SGD parity: our array-level gradient calibrator == ``populace.calibrate``.

:mod:`calibration_paper.sgd` reimplements the gradient calibrator the paper
defends -- Adam on log-weights minimizing capped weighted MAPE -- at the *array*
level, on a ``(A, b, w0)`` system, so the gradient method and the classical
methods (:mod:`calibration_paper.classical`) solve the identical linear problem.
That reimplementation is only trustworthy if it reproduces the package the paper
actually names, ``populace.calibrate``.

**How the parity is pinned.** We build a tiny populace :class:`~populace.frame.Frame`
and :class:`~populace.calibrate.TargetSet`, run ``populace.calibrate.calibrate``,
then read back the problem it *compiled* -- its constraint matrix ``A``, target
vector ``b``, and initial weights ``w0`` -- and run our :func:`sgd_calibrate` on
that same array system with the same seed and the same (default) epochs, learning
rate, and loss cap. Because it is the same objective, the same log-weight
parameterization, and the same torch optimizer, the two must agree to floating
point. This is the array form verified against the Frame form, not a loose
"both roughly calibrate" check.

Skipped unless the ``methods`` extra is installed (torch *and* populace); the
imports below ``importorskip`` both, so the base-install CI never runs -- or
needs -- this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The methods extra: torch (the optimizer) and populace (the reference). Either
# absent => skip the whole module, never fail.
pytest.importorskip("torch", reason="methods extra (torch) not installed")
populace_calibrate = pytest.importorskip(
    "populace.calibrate", reason="methods extra (populace-calibrate) not installed"
)
populace_frame = pytest.importorskip(
    "populace.frame", reason="methods extra (populace-frame) not installed"
)

from calibration_paper.sgd import sgd_calibrate  # noqa: E402

EntitySchema = populace_frame.EntitySchema
Frame = populace_frame.Frame
Weights = populace_frame.Weights
WeightKind = populace_frame.WeightKind
Target = populace_calibrate.Target
TargetSet = populace_calibrate.TargetSet
populace_calibrate_fn = populace_calibrate.calibrate


def _one_person_per_household_frame(
    *, n: int, seed: int
) -> tuple[Frame, np.ndarray, np.ndarray, np.ndarray]:
    """Build a minimal valid populace Frame: one person per household.

    A single person per household makes the household weight vector (what
    ``calibrate`` optimizes) align one-to-one with the per-person measures, so
    the compiled ``A`` is exactly the per-record measure rows -- the simplest
    frame that still exercises the real compiler and solver.

    Args:
        n: Number of households (= persons = length of the weight vector).
        seed: Seed for the record draws and design weights.

    Returns:
        ``(frame, w0, income, is_child)`` -- the frame plus the arrays the
        targets are built from.
    """
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.0, sigma=0.5, size=n)
    is_child = (rng.random(n) < 0.3).astype(np.float64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "income": income,
            "is_child": is_child,
        }
    )
    household = pd.DataFrame({"household_id": np.arange(n)})
    schema = EntitySchema(person_entity="person", group_entities=("household",))
    w0 = rng.lognormal(mean=6.0, sigma=0.3, size=n)
    frame = Frame(
        tables={"person": person, "household": household},
        schema=schema,
        weights={"household": Weights(values=w0, kind=WeightKind.DESIGN)},
    )
    return frame, w0, income, is_child


def _targets(income: np.ndarray, is_child: np.ndarray, w0: np.ndarray) -> TargetSet:
    """Two feasible-but-perturbed sum targets on the person entity.

    Perturbing each off its baseline (``income @ w0`` etc.) guarantees the
    calibrator actually moves the weights, so the parity test exercises the
    optimizer rather than a no-op at ``w0``.
    """
    base_income = float(income @ w0)
    base_children = float(is_child @ w0)
    return TargetSet(
        [
            Target(
                name="total_income",
                entity="person",
                measure="income",
                value=1.15 * base_income,
            ),
            Target(
                name="child_count",
                entity="person",
                measure="is_child",
                value=0.9 * base_children,
            ),
        ]
    )


def test_sgd_matches_populace_calibrate_weights() -> None:
    """Our array SGD reproduces ``populace.calibrate`` on its own compiled system.

    Run populace's calibrate, take the ``(A, b, w0)`` it compiled, run ours on
    it with the same seed, and assert the calibrated weights coincide. Same
    objective + same optimizer => agreement to floating point.
    """
    frame, w0, income, is_child = _one_person_per_household_frame(n=40, seed=0)
    targets = _targets(income, is_child, w0)

    result = populace_calibrate_fn(frame, targets, weight_entity="household", seed=0)
    a = result.problem.matrix
    b = result.problem.target_vector
    w0_compiled = result.initial_weights

    ours = sgd_calibrate(a, b, w0_compiled, seed=0)

    np.testing.assert_allclose(ours.weights, result.weights, rtol=1e-4, atol=1e-3)


def test_sgd_matches_populace_final_loss() -> None:
    """The array form's final capped-MAPE loss matches populace's closing loss.

    populace reports ``final_loss`` (its ``closing_loss``) on the returned
    weights; our :class:`~calibration_paper.sgd.SGDSolution` reports the same
    quantity on the same objective, so they agree.
    """
    frame, w0, income, is_child = _one_person_per_household_frame(n=48, seed=1)
    targets = _targets(income, is_child, w0)

    result = populace_calibrate_fn(frame, targets, weight_entity="household", seed=1)
    ours = sgd_calibrate(
        result.problem.matrix,
        result.problem.target_vector,
        result.initial_weights,
        seed=1,
    )
    assert ours.final_loss == pytest.approx(result.final_loss, rel=1e-4, abs=1e-6)


def test_sgd_bounded_matches_populace_max_weight_ratio() -> None:
    """The hard ratio bound agrees with populace's ``max_weight_ratio`` guard.

    populace clamps ``w <= max_weight_ratio * w0`` after every step (the
    "landmine guard"); our ``max_weight_ratio`` does the same. On the same
    compiled system with the same bound and seed, the bounded solutions match
    and neither exceeds the bound.
    """
    frame, w0, income, is_child = _one_person_per_household_frame(n=40, seed=2)
    # A demanding target that pushes some weights toward the bound.
    base_income = float(income @ w0)
    targets = TargetSet(
        [
            Target(
                name="total_income",
                entity="person",
                measure="income",
                value=1.6 * base_income,
            )
        ]
    )
    bound = 3.0
    result = populace_calibrate_fn(
        frame, targets, weight_entity="household", seed=2, max_weight_ratio=bound
    )
    ours = sgd_calibrate(
        result.problem.matrix,
        result.problem.target_vector,
        result.initial_weights,
        max_weight_ratio=bound,
        seed=2,
    )
    # Neither exceeds the hard bound (both apply the closing float64 cap).
    assert (ours.weights / result.initial_weights).max() <= bound + 1e-9
    assert (result.weights / result.initial_weights).max() <= bound + 1e-6
    np.testing.assert_allclose(ours.weights, result.weights, rtol=1e-4, atol=1e-3)
