"""The calibrator surface: every reweighter under one registry and contract.

Every calibration method the paper compares is registered here as a
:class:`Method`: a key, a family, a human description, the BibTeX citation key
that grounds it, and a lazy ``constructor`` returning a
:class:`CalibratorFn`. The registry is the single source of truth for "what is
compared against what", so the sweep, the tables, and the paper's method list
all read from it -- exactly the pattern the sibling imputation-paper uses for
the fill operator.

**Adapter contract.** A constructor returns a ``calibrate`` callable with the
uniform signature (the calibrate-operator analogue of imputation-paper's
``fit``)::

    calibrate(problem, *, seed=0) -> CalibrationOutcome

where ``problem`` is a :class:`~calibration_paper.problem.CalibrationProblem`
bundling the constraint matrix ``A`` (the candidate frame's per-record target
contributions), the target vector ``b``, and the input weights ``w0`` -- i.e.
issue #1's ``calibrate(frame, targets, weights_in) -> weights_out`` at the
linear-algebra level every calibrator shares. ``seed`` drives any stochastic
method so paired-seed sweeps reproduce. The outcome carries the calibrated
weights plus method-reported convergence, so the sweep can record whether a
method actually hit the surface (a bounded method on an infeasible surface does
not, and says so).

**Lazy imports are load-bearing.** The classical adapters
(:mod:`calibration_paper.classical`) are pure NumPy and always available. The
gradient adapters need torch, and the populace-backed reference adapter needs
``populace-calibrate``; both live behind the ``methods`` extra and are *not*
installed in CI. So neither torch nor populace may be imported at module import
time: each constructor's returned ``calibrate`` imports what it needs inside its
own body. This lets ``list_methods()`` and the registry import (and test) on the
base install alone, while a gradient run raises a clear
:class:`ModuleNotFoundError` only if called without torch -- which the sweep
records as a skip, never a silent drop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from calibration_paper.problem import (
    CalibrationProblem,
    WeightDiagnostics,
    target_relative_errors,
    weight_diagnostics,
)

__all__ = [
    "Method",
    "MethodFamily",
    "CalibratorFn",
    "CalibrationOutcome",
    "REGISTRY",
    "list_methods",
    "get_method",
    "run_method",
    "CLASSICAL_KEYS",
    "GRADIENT_KEYS",
]


class MethodFamily:
    """Registry families, exposed as constants so callers avoid magic strings."""

    #: A classical survey-calibration method (Deville-Sarndal): raking, GREG,
    #: chi-square, entropy, or a bounded variant. Pure NumPy, always available.
    CLASSICAL = "classical"
    #: The gradient calibrator the paper defends (Adam on log-weights) and its
    #: hard-bounded variant. Needs torch (the ``methods`` extra).
    GRADIENT = "gradient"


@dataclass(frozen=True)
class CalibrationOutcome:
    """The output of a calibrator adapter: weights plus what the method reported.

    Attributes:
        weights: The calibrated weight vector ``w``, length ``n_records``.
        converged: Whether the method reports it reproduced the targets to its
            tolerance. ``False`` is a first-class result -- a bounded method on
            an infeasible surface returns its closest feasible weights and says
            so, which is exactly the failure-mode signal the paper studies.
        iterations: Iterations/epochs the method took (for the runtime axis).
        max_scaled_residual: The method's own worst scaled target miss at its
            solution (``max|A@w - b| / max(|b|, 1)``), or ``None`` if the method
            does not report one.
    """

    weights: np.ndarray
    converged: bool
    iterations: int
    max_scaled_residual: float | None = None


class CalibratorFn(Protocol):
    """A calibrator: maps a calibration problem to calibrated weights.

    ``seed`` drives any stochastic method; deterministic methods accept and
    ignore it so the sweep can call every method uniformly.
    """

    def __call__(
        self, problem: CalibrationProblem, *, seed: int = 0
    ) -> CalibrationOutcome: ...


@dataclass(frozen=True)
class Method:
    """One entry in the calibrator surface.

    Attributes:
        key: Stable identifier used in configs, CSVs, and tables.
        family: One of :class:`MethodFamily`.
        description: One-line human description.
        citation_key: BibTeX key in ``paper/bibliography/references.bib`` that
            grounds the method.
        guarantees_positive: Whether the method's weights are non-negative by
            construction (raking, logit, and the gradient methods: yes; GREG /
            unbounded chi-square: no). The sweep checks this invariant so a
            regression that broke a positivity guarantee is caught.
        respects_bounds: The ``(L, U)`` g-weight bounds the method holds by
            construction, or ``None`` for an unbounded method. Checked by the
            adapter tests.
        constructor: Zero-argument callable returning the method's
            :class:`CalibratorFn`. The returned callable imports any heavy
            dependency lazily.
    """

    key: str
    family: str
    description: str
    citation_key: str
    guarantees_positive: bool
    respects_bounds: tuple[float, float] | None
    constructor: Callable[[], CalibratorFn]


# ---------------------------------------------------------------------------
# Classical adapters (pure NumPy; always available)
# ---------------------------------------------------------------------------


def _classical_constructor(
    distance: str, *, bounds: tuple[float, float] | None = None
) -> CalibratorFn:
    """Adapter for a classical Deville-Sarndal calibration distance.

    Args:
        distance: ``"linear"`` (GREG), ``"raking"`` (= entropy/KL),
            ``"chi_square"`` (GREG unbounded, truncated-linear with ``bounds``),
            or ``"logit"`` (requires ``bounds``).
        bounds: ``(L, U)`` g-weight bounds where the method uses them.

    Returns:
        A :class:`CalibratorFn`.
    """

    def calibrate(
        problem: CalibrationProblem, *, seed: int = 0  # noqa: ARG001 - deterministic
    ) -> CalibrationOutcome:
        from calibration_paper.classical import calibrate_distance

        solution = calibrate_distance(
            distance,
            problem.matrix,
            problem.target,
            problem.initial_weights,
            bounds=bounds,
        )
        return CalibrationOutcome(
            weights=solution.weights,
            converged=solution.converged,
            iterations=solution.iterations,
            max_scaled_residual=solution.max_scaled_residual,
        )

    return calibrate


# ---------------------------------------------------------------------------
# Gradient adapters (torch; the ``methods`` extra)
# ---------------------------------------------------------------------------


def _sgd_constructor(*, max_weight_ratio: float | None = None) -> CalibratorFn:
    """Adapter for the gradient calibrator (Adam on log-weights).

    Args:
        max_weight_ratio: The hard per-record ratio bound of the ``sgd_bounded``
            variant, or ``None`` for the unbounded ``sgd`` method.

    Returns:
        A :class:`CalibratorFn`. Its call imports torch lazily, so the registry
        imports without torch and a run without it is a recorded skip.
    """

    def calibrate(
        problem: CalibrationProblem, *, seed: int = 0
    ) -> CalibrationOutcome:
        from calibration_paper.sgd import sgd_calibrate

        solution = sgd_calibrate(
            problem.matrix,
            problem.target,
            problem.initial_weights,
            max_weight_ratio=max_weight_ratio,
            seed=seed,
        )
        # The gradient method has no hard convergence flag; report whether its
        # final loss reproduced every target within 0.5% scaled error, the same
        # bar the classical Newton tolerance targets in practice.
        residuals = target_relative_errors(problem, solution.weights)
        max_scaled = float(np.max(np.abs(residuals)))
        return CalibrationOutcome(
            weights=solution.weights,
            converged=max_scaled < 5e-3,
            iterations=solution.epochs,
            max_scaled_residual=max_scaled,
        )

    return calibrate


#: The calibrator surface. Ordered classical (raking -> GREG -> chi-square ->
#: entropy -> logit-bounded) then gradient (SGD -> SGD-bounded), matching
#: PLAN.md's method table.
REGISTRY: dict[str, Method] = {
    # --- Classical (survey-statistics standard) ---
    "raking": Method(
        key="raking",
        family=MethodFamily.CLASSICAL,
        description=(
            "Raking / iterative proportional fitting: multiplicative g-weights "
            "g = exp(x' lambda), the margins-only classical baseline. Weights "
            "positive by construction."
        ),
        citation_key="deville1992calibration",
        guarantees_positive=True,
        respects_bounds=None,
        constructor=lambda: _classical_constructor("raking"),
    ),
    "greg": Method(
        key="greg",
        family=MethodFamily.CLASSICAL,
        description=(
            "GREG / linear calibration (Deville-Sarndal linear distance): "
            "closed-form g = 1 + x' lambda. Can produce negative weights -- the "
            "incidence is reported."
        ),
        citation_key="deville1992calibration",
        guarantees_positive=False,
        respects_bounds=None,
        constructor=lambda: _classical_constructor("linear"),
    ),
    "chi_square": Method(
        key="chi_square",
        family=MethodFamily.CLASSICAL,
        description=(
            "Chi-square distance calibration, unbounded (= GREG): quadratic "
            "distance G(g) = (g-1)^2 / 2. The bounded variant is 'chi_square_bounded'."
        ),
        citation_key="deville1992calibration",
        guarantees_positive=False,
        respects_bounds=None,
        constructor=lambda: _classical_constructor("chi_square"),
    ),
    "chi_square_bounded": Method(
        key="chi_square_bounded",
        family=MethodFamily.CLASSICAL,
        description=(
            "Chi-square distance with hard g-weight bounds (truncated linear, "
            "g in [0.2, 5]): the standard remedy for GREG's extreme/negative "
            "weights. Exact targets only when feasible."
        ),
        citation_key="deville1992calibration",
        guarantees_positive=True,
        respects_bounds=(0.2, 5.0),
        constructor=lambda: _classical_constructor("chi_square", bounds=(0.2, 5.0)),
    ),
    "entropy": Method(
        key="entropy",
        family=MethodFamily.CLASSICAL,
        description=(
            "Entropy balancing / exponential tilting: the minimum-KL reweighting "
            "from the design weights. Mathematically identical to raking; kept as "
            "a separate key because the literature names it separately."
        ),
        citation_key="hainmueller2012entropy",
        guarantees_positive=True,
        respects_bounds=None,
        constructor=lambda: _classical_constructor("raking"),
    ),
    "logit_bounded": Method(
        key="logit_bounded",
        family=MethodFamily.CLASSICAL,
        description=(
            "Bounded calibration via the Deville-Sarndal logit calibration "
            "function: g-weights smoothly confined to (0.2, 5). The classical "
            "hard-bounded reweighter."
        ),
        citation_key="deville1992calibration",
        guarantees_positive=True,
        respects_bounds=(0.2, 5.0),
        constructor=lambda: _classical_constructor("logit", bounds=(0.2, 5.0)),
    ),
    # --- Gradient (the calibrator the paper defends) ---
    "sgd": Method(
        key="sgd",
        family=MethodFamily.GRADIENT,
        description=(
            "Gradient-descent calibration (populace-calibrate): Adam on "
            "log-weights minimizing capped fixed-scale weighted MAPE. Weights "
            "positive by construction; a soft loss, so no hard target guarantee."
        ),
        citation_key="populace2026",
        guarantees_positive=True,
        respects_bounds=None,
        constructor=lambda: _sgd_constructor(max_weight_ratio=None),
    ),
    "sgd_bounded": Method(
        key="sgd_bounded",
        family=MethodFamily.GRADIENT,
        description=(
            "Gradient calibration with a hard per-record weight-ratio bound "
            "(max_weight_ratio=5): the landmine-guard variant that clamps the "
            "weight ratio after every step."
        ),
        citation_key="populace2026",
        guarantees_positive=True,
        respects_bounds=(0.0, 5.0),
        constructor=lambda: _sgd_constructor(max_weight_ratio=5.0),
    ),
}

#: Keys grouped by family, for the sweep and the paper's method list.
CLASSICAL_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.family == MethodFamily.CLASSICAL
)
GRADIENT_KEYS: tuple[str, ...] = tuple(
    k for k, m in REGISTRY.items() if m.family == MethodFamily.GRADIENT
)


def list_methods(family: str | None = None) -> list[str]:
    """Return registry keys, optionally filtered to one :class:`MethodFamily`."""
    if family is None:
        return list(REGISTRY)
    return [k for k, m in REGISTRY.items() if m.family == family]


def get_method(key: str) -> Method:
    """Return the :class:`Method` for ``key`` (raises ``KeyError`` if unknown)."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown method {key!r}. Registered methods: {sorted(REGISTRY)}."
        ) from None


@dataclass(frozen=True)
class MethodResult:
    """A method run on a problem: the outcome plus computed diagnostics.

    Attributes:
        key: The method's registry key.
        outcome: The adapter's :class:`CalibrationOutcome`.
        diagnostics: Weight-quality diagnostics
            (:class:`~calibration_paper.problem.WeightDiagnostics`) on the
            calibrated weights.
        relative_errors: Signed per-target relative errors at the solution.
    """

    key: str
    outcome: CalibrationOutcome
    diagnostics: WeightDiagnostics
    relative_errors: np.ndarray


def run_method(
    key: str, problem: CalibrationProblem, *, seed: int = 0
) -> MethodResult:
    """Run one registered method on ``problem`` and compute its diagnostics.

    The inner cell of the sweep: construct the method's adapter, calibrate, then
    score the weights with the weight diagnostics and per-target relative errors.
    Method-agnostic -- it only knows the :class:`CalibratorFn` contract -- so the
    classical and gradient methods all run through the same path.

    Args:
        key: A key into :data:`REGISTRY`.
        problem: The calibration problem to solve.
        seed: Seed passed to the (possibly stochastic) method.

    Returns:
        A :class:`MethodResult`.
    """
    method = get_method(key)
    calibrate = method.constructor()
    outcome = calibrate(problem, seed=seed)
    diagnostics = weight_diagnostics(outcome.weights, problem.initial_weights)
    errors = target_relative_errors(problem, outcome.weights)
    return MethodResult(
        key=key,
        outcome=outcome,
        diagnostics=diagnostics,
        relative_errors=errors,
    )
