"""Classical survey calibration in the Deville-Sarndal (1992) framework.

Classical calibration finds weights ``w_k = w0_k * g_k`` closest to the design
weights ``w0`` under a distance ``G`` while reproducing the targets exactly:

    minimize   sum_k w0_k * G(g_k)
    subject to sum_k w_k * x_k = t                       (the calibration equations)

where ``x_k`` is record ``k``'s column of the constraint matrix ``A`` (so
``sum_k w_k x_k = A @ w``) and ``t`` is the target vector ``b``. The distance
``G`` fixes the *method*; its derivative's inverse ``g = F(u)`` gives the
calibration function, and the g-weights have the single-index form
``g_k = F(x_k' @ lambda)`` for a vector of Lagrange multipliers ``lambda`` (one
per target). Newton's method on the ``n_targets`` calibration equations solves
for ``lambda``; the Jacobian is ``-sum_k w0_k F'(u_k) x_k x_k'``.

The four distances give the paper's classical methods
(Deville & Sarndal 1992, Table 1):

* **linear** ``G(g) = (g-1)^2 / 2``  ->  ``F(u) = 1 + u``. Closed form (one
  Newton step from ``lambda = 0``); the GREG estimator. Weights can go negative.
* **raking** ``G(g) = g log g - g + 1``  ->  ``F(u) = exp(u)``. The
  multiplicative / iterative-proportional-fitting method; equivalently the
  minimum-KL (exponential-tilting / entropy-balancing) reweighting. Weights are
  strictly positive by construction.
* **truncated linear** ``F(u) = clip(1 + u, L, U)`` -- the linear method with
  hard g-weight bounds ``[L, U]`` (the chi-square-with-bounds method). Bounded
  weight ratios; exact target reproduction only when feasible.
* **logit** the Deville-Sarndal logit calibration function mapping onto
  ``[L, U]``:  ``F(u) = (L(U-1) + U(1-L) e^{A u}) / ((U-1) + (1-L) e^{A u})``
  with ``A = (U-L) / ((1-L)(U-1))``. Smoothly bounded g-weights in ``(L, U)``.

Everything is pure NumPy/SciPy so the classical adapters run on the base install
(no R, no torch). ``tests/test_r_parity.py`` checks these against R ``survey``'s
``calibrate``/``grake`` on small cases when rpy2 is installed, so the comparison
is grounded in the field's actual tool, not this reimplementation alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

__all__ = [
    "CalibrationSolution",
    "raking_weights",
    "linear_weights",
    "chi_square_weights",
    "logit_weights",
    "calibrate_distance",
]

#: Default Newton tolerance on the max absolute *scaled* calibration-equation
#: residual ``max|A@w - b| / max(|b|, 1)``. Below this the targets are hit.
_DEFAULT_TOL = 1e-9

#: Default Newton iteration cap. The smooth methods converge quadratically; the
#: bounded methods can plateau at the feasibility boundary, which is reported
#: (``converged=False``) rather than iterated forever.
_DEFAULT_MAX_ITER = 100

#: Ridge added to the Newton Jacobian's diagonal when it is singular or
#: ill-conditioned (collinear targets). Tiny relative to the Jacobian scale, it
#: only regularizes the solve direction; it does not change the fixed point.
_JACOBIAN_RIDGE = 1e-10


@dataclass(frozen=True)
class CalibrationSolution:
    """The result of a classical calibration solve.

    Attributes:
        weights: The calibrated weights ``w = w0 * g``.
        g_weights: The g-weights ``g = w / w0`` (the calibration ratios).
        lambdas: The Lagrange multipliers (one per target) at the solution.
        iterations: Newton iterations taken.
        converged: Whether the scaled calibration-equation residual fell below
            the tolerance. ``False`` means the targets were not reproduced to
            tolerance -- expected for a deliberately infeasible surface or a
            bounded method whose bounds preclude an exact hit.
        max_scaled_residual: The final ``max|A@w - b| / max(|b|, 1)`` -- the
            worst target's scaled miss (``0`` at exact reproduction).
    """

    weights: np.ndarray
    g_weights: np.ndarray
    lambdas: np.ndarray
    iterations: int
    converged: bool
    max_scaled_residual: float


def _as_dense_columns(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Return ``A`` as a dense ``(n_targets, n_records)`` float64 array.

    The classical Newton solve forms the ``n_targets x n_targets`` Jacobian by
    weighting the columns of ``A``; it materializes ``A`` densely. That is fine
    for the paper's classical-method regime (thousands of targets, the Jacobian
    is the cost); the gradient methods keep ``A`` sparse for the national pool.
    """
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _newton_calibrate(
    matrix: np.ndarray,
    target: np.ndarray,
    initial_weights: np.ndarray,
    calibration_function,
    calibration_derivative,
    *,
    tol: float,
    max_iter: int,
) -> CalibrationSolution:
    """Newton's method for the calibration equations of a given ``F, F'``.

    Solves ``sum_k w0_k F(u_k) x_k = t`` for ``lambda`` where ``u_k = x_k' @
    lambda`` and ``x_k`` is column ``k`` of ``A``. The update is
    ``lambda += J^{-1} (t - A w)`` with ``J = A diag(w0 F'(u)) A'`` (the
    ``n_targets x n_targets`` weighted Gram matrix).

    Args:
        matrix: Dense ``A`` of shape ``(n_targets, n_records)``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        calibration_function: ``F(u) -> g`` applied elementwise to ``u``.
        calibration_derivative: ``F'(u)`` applied elementwise to ``u``.
        tol: Stop when the max scaled calibration residual falls below this.
        max_iter: Newton iteration cap.

    Returns:
        A :class:`CalibrationSolution`. ``converged`` reflects the final
        residual against ``tol``; the last iterate is always returned.
    """
    a = matrix
    w0 = initial_weights
    n_targets = a.shape[0]
    scale = float(max(np.max(np.abs(target)), 1.0))

    lambdas = np.zeros(n_targets, dtype=np.float64)
    best_lambdas = lambdas.copy()
    best_residual = np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        u = a.T @ lambdas
        g = calibration_function(u)
        w = w0 * g
        residual = target - a @ w
        max_scaled = float(np.max(np.abs(residual)) / scale)
        if max_scaled < best_residual:
            best_residual = max_scaled
            best_lambdas = lambdas.copy()
        if max_scaled < tol:
            break
        # Jacobian J = A diag(w0 F'(u)) A' (n_targets x n_targets).
        weighted = w0 * calibration_derivative(u)
        jacobian = (a * weighted) @ a.T
        # Regularize a singular/ill-conditioned Jacobian (collinear targets)
        # rather than raising: the ridge nudges the step direction only.
        ridge = _JACOBIAN_RIDGE * float(np.trace(jacobian)) / max(n_targets, 1)
        jacobian_reg = jacobian + ridge * np.eye(n_targets)
        try:
            step = np.linalg.solve(jacobian_reg, residual)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this rare
            step = np.linalg.lstsq(jacobian_reg, residual, rcond=None)[0]
        lambdas = lambdas + step

    # Report the best iterate seen (Newton on bounded methods can overshoot then
    # come back; the best residual is the honest summary).
    u = a.T @ best_lambdas
    g = calibration_function(u)
    w = w0 * g
    max_scaled = float(np.max(np.abs(target - a @ w)) / scale)
    return CalibrationSolution(
        weights=w,
        g_weights=g,
        lambdas=best_lambdas,
        iterations=iterations,
        converged=max_scaled < tol,
        max_scaled_residual=max_scaled,
    )


def linear_weights(
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> CalibrationSolution:
    """GREG / linear calibration: ``F(u) = 1 + u`` (Deville-Sarndal linear).

    The generalized regression estimator's weights. The calibration function is
    affine, so a single Newton step from ``lambda = 0`` is exact -- the g-weights
    are the closed form ``g = 1 + A' (A diag(w0) A')^{-1} (t - A w0) / ...``
    (the code takes one Newton step, which for a linear ``F`` reaches the exact
    solution). Weights can be negative when a target pulls a record's g-weight
    below zero; :func:`~calibration_paper.problem.weight_diagnostics` reports the
    incidence, as the paper requires.

    Args:
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        tol: Residual tolerance (a linear ``F`` typically hits it in one step).
        max_iter: Newton cap.

    Returns:
        A :class:`CalibrationSolution`.
    """
    a = _as_dense_columns(matrix)
    return _newton_calibrate(
        a,
        np.asarray(target, dtype=np.float64),
        np.asarray(initial_weights, dtype=np.float64),
        calibration_function=lambda u: 1.0 + u,
        calibration_derivative=lambda u: np.ones_like(u),
        tol=tol,
        max_iter=max_iter,
    )


def raking_weights(
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> CalibrationSolution:
    """Raking / multiplicative calibration: ``F(u) = exp(u)`` (minimum KL).

    The iterative-proportional-fitting method, equivalently the reweighting that
    minimizes Kullback-Leibler divergence from the design weights subject to the
    calibration equations -- i.e. exponential tilting / entropy balancing. The
    g-weights ``g = exp(A' lambda)`` are strictly positive, so raked weights are
    always positive (its practical advantage over GREG). Solved by Newton on
    ``lambda`` (equivalent to, and faster than, cyclic IPF for a general design
    matrix).

    Args:
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        tol: Residual tolerance.
        max_iter: Newton cap.

    Returns:
        A :class:`CalibrationSolution` with strictly positive weights.
    """
    a = _as_dense_columns(matrix)
    # Clip the exponent for numerical safety; exp overflow on a wild Newton step
    # would poison the Jacobian. The clip is wide enough never to bind at a
    # sensible solution (g in [e^-30, e^30]).
    def f(u: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(u, -30.0, 30.0))

    return _newton_calibrate(
        a,
        np.asarray(target, dtype=np.float64),
        np.asarray(initial_weights, dtype=np.float64),
        calibration_function=f,
        calibration_derivative=f,  # d/du exp(u) = exp(u)
        tol=tol,
        max_iter=max_iter,
    )


def chi_square_weights(
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> CalibrationSolution:
    """Chi-square distance calibration, optionally with hard g-weight bounds.

    The chi-square distance ``G(g) = (g-1)^2 / 2`` gives the same calibration
    function as :func:`linear_weights` (``F(u) = 1 + u``) -- unbounded chi-square
    *is* GREG. Passing ``bounds=(L, U)`` switches to the **truncated linear**
    method ``F(u) = clip(1 + u, L, U)``: the g-weights are held inside ``[L, U]``,
    the standard remedy for GREG's negative/extreme weights. With bounds the
    targets are reproduced only when feasible; otherwise the solve reports
    ``converged=False`` and returns the closest bounded iterate.

    Args:
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        bounds: ``(L, U)`` hard g-weight bounds (``L < 1 < U``), or ``None`` for
            the unbounded (GREG-equivalent) chi-square.
        tol: Residual tolerance.
        max_iter: Newton cap.

    Returns:
        A :class:`CalibrationSolution`.

    Raises:
        ValueError: If ``bounds`` is given with ``L >= U`` or ``L >= 1`` or
            ``U <= 1`` (the design weight's ratio 1 must be interior).
    """
    a = _as_dense_columns(matrix)
    b = np.asarray(target, dtype=np.float64)
    w0 = np.asarray(initial_weights, dtype=np.float64)
    if bounds is None:
        return linear_weights(a, b, w0, tol=tol, max_iter=max_iter)

    lower, upper = float(bounds[0]), float(bounds[1])
    _validate_bounds(lower, upper)

    def f(u: np.ndarray) -> np.ndarray:
        return np.clip(1.0 + u, lower, upper)

    def f_prime(u: np.ndarray) -> np.ndarray:
        # 1 on the interior, 0 where the clip binds. A record whose g-weight is
        # pinned to a bound contributes nothing to the Jacobian -- it is frozen.
        g = 1.0 + u
        return ((g > lower) & (g < upper)).astype(np.float64)

    return _newton_calibrate(
        a, b, w0, calibration_function=f, calibration_derivative=f_prime,
        tol=tol, max_iter=max_iter,
    )


def logit_weights(
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    bounds: tuple[float, float],
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> CalibrationSolution:
    """Logit bounded calibration (Deville-Sarndal logit calibration function).

    The g-weights are smoothly confined to the open interval ``(L, U)`` by the
    logit calibration function

        ``F(u) = (L(U-1) + U(1-L) e^{A u}) / ((U-1) + (1-L) e^{A u})``,
        ``A = (U - L) / ((1 - L)(U - 1))``,

    which is the ``[L, U]``-bounded analogue of raking: ``F(0) = 1`` (the design
    weight is unchanged at ``lambda = 0``), ``F`` is increasing, and
    ``F(u) -> L`` / ``U`` as ``u -> -inf`` / ``+inf``. This is R ``survey``'s
    ``calfun="logit"``. Unlike truncated-linear the bounds are never exactly
    attained, so the g-weights stay strictly interior and the derivative never
    vanishes.

    Args:
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        bounds: ``(L, U)`` with ``L < 1 < U`` -- the open interval the g-weights
            are confined to.
        tol: Residual tolerance.
        max_iter: Newton cap.

    Returns:
        A :class:`CalibrationSolution` with g-weights strictly inside ``(L, U)``.

    Raises:
        ValueError: If ``L >= U`` or ``L >= 1`` or ``U <= 1``.
    """
    a = _as_dense_columns(matrix)
    b = np.asarray(target, dtype=np.float64)
    w0 = np.asarray(initial_weights, dtype=np.float64)
    lower, upper = float(bounds[0]), float(bounds[1])
    _validate_bounds(lower, upper)
    coef = (upper - lower) / ((1.0 - lower) * (upper - 1.0))

    def f(u: np.ndarray) -> np.ndarray:
        e = np.exp(np.clip(coef * u, -30.0, 30.0))
        numerator = lower * (upper - 1.0) + upper * (1.0 - lower) * e
        denominator = (upper - 1.0) + (1.0 - lower) * e
        return numerator / denominator

    def f_prime(u: np.ndarray) -> np.ndarray:
        # F'(u): quotient rule on F above. Both parts share e = exp(coef u).
        e = np.exp(np.clip(coef * u, -30.0, 30.0))
        numerator = lower * (upper - 1.0) + upper * (1.0 - lower) * e
        denominator = (upper - 1.0) + (1.0 - lower) * e
        d_num = coef * upper * (1.0 - lower) * e
        d_den = coef * (1.0 - lower) * e
        return (d_num * denominator - numerator * d_den) / denominator**2

    return _newton_calibrate(
        a, b, w0, calibration_function=f, calibration_derivative=f_prime,
        tol=tol, max_iter=max_iter,
    )


def _validate_bounds(lower: float, upper: float) -> None:
    """Validate ``(L, U)`` g-weight bounds: ``L < 1 < U``."""
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"bounds must be finite, got ({lower}, {upper}).")
    if lower >= upper:
        raise ValueError(f"bounds must satisfy L < U, got L={lower}, U={upper}.")
    if lower >= 1.0 or upper <= 1.0:
        raise ValueError(
            f"bounds must bracket the design ratio 1 (L < 1 < U), got "
            f"L={lower}, U={upper}."
        )


#: The classical calibration distances, keyed by the name used in the registry
#: and R ``survey``'s ``calfun``. ``chi_square`` is ``linear`` unbounded; the
#: distinction is whether ``bounds`` is passed.
def calibrate_distance(
    distance: str,
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> CalibrationSolution:
    """Dispatch to the classical calibration solver for ``distance``.

    Args:
        distance: One of ``"linear"``, ``"raking"``, ``"chi_square"``,
            ``"logit"``. ``"linear"`` and ``"raking"`` ignore ``bounds``;
            ``"chi_square"`` uses them if given (truncated linear) and is GREG
            without; ``"logit"`` requires them.
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        bounds: ``(L, U)`` g-weight bounds where the method uses them.
        tol: Residual tolerance.
        max_iter: Newton cap.

    Returns:
        A :class:`CalibrationSolution`.

    Raises:
        ValueError: On an unknown ``distance``, or ``"logit"`` without bounds.
    """
    if distance == "linear":
        return linear_weights(matrix, target, initial_weights, tol=tol, max_iter=max_iter)
    if distance == "raking":
        return raking_weights(matrix, target, initial_weights, tol=tol, max_iter=max_iter)
    if distance == "chi_square":
        return chi_square_weights(
            matrix, target, initial_weights, bounds=bounds, tol=tol, max_iter=max_iter
        )
    if distance == "logit":
        if bounds is None:
            raise ValueError("logit calibration requires bounds=(L, U).")
        return logit_weights(
            matrix, target, initial_weights, bounds=bounds, tol=tol, max_iter=max_iter
        )
    raise ValueError(
        f"Unknown calibration distance {distance!r}; expected one of "
        "'linear', 'raking', 'chi_square', 'logit'."
    )
