"""The gradient calibrator: Adam on log-weights, capped weighted MAPE.

This is the array-level form of the calibrator the paper defends -- the exact
objective and algorithm ``populace.calibrate`` uses, run directly on a
:class:`~calibration_paper.problem.CalibrationProblem`'s ``(A, b, w0)`` rather
than a populace Frame. Fixing it at the array level means the gradient method
and the classical methods (:mod:`calibration_paper.classical`) solve the
*identical* linear problem, which is the whole point of the comparison.

The objective is capped fixed-scale weighted MAPE

    ``mean_i min(|(A @ w - b)_i / s_i|, cap)``,   ``s_i = max(|b_i|, 1)``,

minimized by Adam over the **log-weights** ``log w`` (so ``w = exp(log w) > 0``
by construction -- the gradient method's positivity guarantee, matching raking
and unlike GREG). The bounded variant adds a hard per-record ratio clamp
``w <= max_weight_ratio * w0`` after every step (populace-calibrate's "landmine
guard"): a rare high-value, near-zero-weight record can never have its weight
detonate on reweight.

The formula, the log-weight parameterization, the fixed target-defined scale,
the 1000% contribution cap, and the post-step ratio clamp are all copied from
``populace.calibrate.solve`` so this is a faithful reimplementation, not a new
method. ``tests/test_sgd_parity.py`` checks it reproduces
``populace.calibrate.calibrate`` on a small synthetic Frame when the ``methods``
extra is installed, so the array form is verified against the package the paper
names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

__all__ = [
    "SGDSolution",
    "sgd_calibrate",
    "capped_weighted_mape",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_TARGET_LOSS_CAP",
]

#: Adam epochs. Matches populace-calibrate's default so the array form and the
#: package agree without a per-call override.
DEFAULT_EPOCHS = 256

#: Adam learning rate on the log-weights. Capped MAPE has a near-constant
#: gradient away from zero, so this is deliberately small (populace-calibrate's
#: default) to avoid oscillating around feasible targets.
DEFAULT_LEARNING_RATE = 0.02

#: Per-target contribution cap for weighted MAPE: a target contributes at most a
#: 1000%-scaled miss to the objective (populace-calibrate's default).
DEFAULT_TARGET_LOSS_CAP = 10.0


@dataclass(frozen=True)
class SGDSolution:
    """The result of a gradient calibration solve.

    Attributes:
        weights: The calibrated weights ``w`` (strictly positive).
        g_weights: The g-weights ``w / w0``.
        loss_trajectory: The capped-weighted-MAPE loss at each epoch's start
            (length ``epochs``); the first entry is the input-weight loss.
        final_loss: The loss re-evaluated on the returned weights (after the
            closing ratio clamp), so it describes the shipped vector.
        epochs: Number of Adam steps taken.
        max_weight_ratio: The hard ratio bound applied (``None`` if unbounded).
    """

    weights: np.ndarray
    g_weights: np.ndarray
    loss_trajectory: np.ndarray
    final_loss: float
    epochs: int
    max_weight_ratio: float | None


def capped_weighted_mape(
    estimates: np.ndarray,
    target: np.ndarray,
    *,
    scales: np.ndarray | None = None,
    cap: float = DEFAULT_TARGET_LOSS_CAP,
) -> float:
    """The paper's calibration loss in NumPy: ``mean min(|(est-b)/s|, cap)``.

    The single scalar every scorer and the Adam objective share. The scale ``s``
    defaults to ``max(|b|, 1)`` -- the target's own value, or one unit for a
    zero-valued target -- so the loss is a scaled relative error, and each
    target's contribution is capped at ``cap`` (default 1000%).

    Args:
        estimates: The achieved aggregates ``A @ w``.
        target: The target vector ``b``.
        scales: Optional per-target denominators; default ``max(|b|, 1)``.
        cap: Per-target cap on the scaled absolute miss.

    Returns:
        The mean capped scaled absolute error.

    Raises:
        ValueError: On a shape mismatch or non-finite inputs (a NaN estimate is
            a bug, not a large miss).
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if estimates.shape != target.shape:
        raise ValueError(
            f"estimates and target must align, got {estimates.shape} vs "
            f"{target.shape}."
        )
    if not (np.isfinite(estimates).all() and np.isfinite(target).all()):
        raise ValueError("capped_weighted_mape requires finite inputs.")
    if scales is None:
        scales = np.maximum(np.abs(target), 1.0)
    loss = np.minimum(np.abs((estimates - target) / scales), cap)
    return float(loss.mean())


def _torch_matrix(matrix, torch):
    """Torch operator for ``A``: sparse CSR when the matrix is sparse, else dense.

    Mirrors populace-calibrate: a sparse ``A`` (the national hierarchical
    surface) stays sparse so a step is SpMM, not a dense 1 GB multiply.
    """
    if sparse.issparse(matrix):
        csr = matrix.tocsr().astype(np.float32)
        return torch.sparse_csr_tensor(
            torch.from_numpy(np.asarray(csr.indptr, dtype=np.int64)),
            torch.from_numpy(np.asarray(csr.indices, dtype=np.int64)),
            torch.from_numpy(csr.data),
            size=csr.shape,
        )
    return torch.tensor(np.asarray(matrix, dtype=np.float64), dtype=torch.float32)


def _apply(matrix, weights, torch):
    """``A @ w`` for a dense or sparse-CSR torch operator."""
    if matrix.layout == torch.sparse_csr:
        return (matrix @ weights.unsqueeze(1)).squeeze(1)
    return matrix @ weights


def sgd_calibrate(
    matrix: np.ndarray | sparse.spmatrix,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    max_weight_ratio: float | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    target_loss_cap: float = DEFAULT_TARGET_LOSS_CAP,
    seed: int = 0,
) -> SGDSolution:
    """Calibrate by Adam on log-weights against capped weighted MAPE.

    The gradient calibrator the paper defends, at the array level. Optimizes
    ``log w`` (so ``w > 0``) to minimize ``mean min(|(A@w - b)/s|, cap)`` with
    ``s = max(|b|, 1)``. When ``max_weight_ratio`` is given, the log-weights are
    clamped after every step so ``w <= max_weight_ratio * w0`` -- the hard bound
    of the ``sgd_bounded`` method -- and a closing float64 clamp makes the bound
    exact on the returned vector.

    Args:
        matrix: The constraint matrix ``A``.
        target: The target vector ``b``.
        initial_weights: The design weights ``w0`` (strictly positive).
        max_weight_ratio: Hard per-record ratio bound (``sgd_bounded``), or
            ``None`` for the unbounded gradient method (``sgd``).
        epochs: Adam steps.
        learning_rate: Adam learning rate on the log-weights.
        target_loss_cap: Per-target cap on the scaled absolute miss.
        seed: Torch RNG seed (for reproducibility; the objective is
            deterministic given it).

    Returns:
        An :class:`SGDSolution`.

    Raises:
        ModuleNotFoundError: If torch is not installed (the ``methods`` extra).
        ValueError: On non-positive ``w0``, non-positive ``epochs``, or a
            ``max_weight_ratio`` that is not positive.
    """
    import torch

    b = np.asarray(target, dtype=np.float64)
    w0 = np.asarray(initial_weights, dtype=np.float64)
    if (w0 <= 0.0).any():
        raise ValueError("initial_weights must be strictly positive.")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if max_weight_ratio is not None and not (max_weight_ratio > 0):
        raise ValueError(
            f"max_weight_ratio must be positive, got {max_weight_ratio}."
        )

    torch.manual_seed(seed)
    matrix_t = _torch_matrix(matrix, torch)
    target_t = torch.tensor(b, dtype=torch.float32)
    scales_t = torch.tensor(np.maximum(np.abs(b), 1.0), dtype=torch.float32)
    log_w = torch.tensor(np.log(w0), dtype=torch.float32, requires_grad=True)
    upper = (
        torch.tensor(max_weight_ratio * w0, dtype=torch.float32)
        if max_weight_ratio is not None
        else None
    )
    optimizer = torch.optim.Adam([log_w], lr=learning_rate)

    trajectory = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = torch.exp(log_w)
        estimate = _apply(matrix_t, weights, torch)
        scaled = torch.abs((estimate - target_t) / scales_t)
        loss = torch.clamp(scaled, max=float(target_loss_cap)).mean()
        trajectory[epoch] = float(loss.item())
        loss.backward()
        optimizer.step()
        if upper is not None:
            with torch.no_grad():
                # The landmine guard: clamp log-weights so exp(log_w) never
                # exceeds max_weight_ratio * w0.
                log_w.clamp_(max=torch.log(upper))

    with torch.no_grad():
        final = torch.exp(log_w).detach().numpy().astype(np.float64)
    if max_weight_ratio is not None:
        # Make the bound exact on the returned vector (float32 exp can overshoot
        # by an epsilon; downstream code may assert the bound).
        final = np.minimum(final, max_weight_ratio * w0)

    final_estimates = np.asarray(matrix @ final, dtype=np.float64).ravel()
    final_loss = capped_weighted_mape(
        final_estimates, b, scales=np.maximum(np.abs(b), 1.0), cap=target_loss_cap
    )
    return SGDSolution(
        weights=final,
        g_weights=final / w0,
        loss_trajectory=trajectory,
        final_loss=final_loss,
        epochs=epochs,
        max_weight_ratio=max_weight_ratio,
    )
