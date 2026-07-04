"""R-parity: the pure-NumPy classical solvers match R ``survey``'s ``calibrate``.

The paper compares the gradient calibrator against *the field's actual tools*.
:mod:`calibration_paper.classical` reimplements Deville-Sarndal calibration in
NumPy so the sweep runs without an R toolchain, but a reimplementation is only
credible if it reproduces the reference implementation. This test grounds it
against R's ``survey`` package (Lumley), the de facto standard for calibration
in official statistics.

**Design: a subprocess, not an ABI.** The bridge to R is the small driver
``scripts/survey_calibrate.R`` run via ``Rscript`` with CSV in/out -- deliberately
*not* rpy2, whose in-process R embedding couples the test to a compiled ABI and
a matched R/rpy2/NumPy build. A subprocess needs only an ``Rscript`` on ``PATH``
with ``survey`` installed; when either is absent the whole module skips (it never
fails for a missing optional tool), so CI on the base install is unaffected and a
developer with R gets the grounding for free.

**What "parity" means numerically.** ``survey::calibrate`` stops at its own
``epsilon`` (default ``1e-7`` on the scaled calibration residual); our Newton
solver stops at ``1e-9``, so on a feasible problem *both* reproduce the targets
and their g-weights agree to roughly the looser of the two tolerances. We assert
agreement to ``1e-6`` on the g-weights -- comfortably inside R's own convergence
epsilon, i.e. the two implementations land on the same fixed point to the
precision R itself resolves. (In practice they agree far tighter: raking and
logit to ~1e-14, GREG/linear to ~1e-10.)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from calibration_paper.classical import (
    linear_weights,
    logit_weights,
    raking_weights,
)
from calibration_paper.synthetic import feasible_problem

#: The R driver shipped in the repo; the subprocess parity oracle.
_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "survey_calibrate.R"

#: g-weight agreement tolerance vs R (inside R's own ``epsilon`` of 1e-7).
_PARITY_RTOL = 1e-6
_PARITY_ATOL = 1e-6


def _rscript() -> str | None:
    """Path to ``Rscript`` if it is on ``PATH``, else ``None``."""
    return shutil.which("Rscript")


def _survey_available(rscript: str) -> bool:
    """Whether R's ``survey`` package loads (needed by the driver)."""
    probe = subprocess.run(
        [rscript, "-e", "cat(as.character(suppressMessages(require(survey))))"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "TRUE"


def _skip_reason() -> str | None:
    """Return why the R-parity suite must skip, or ``None`` if it can run."""
    rscript = _rscript()
    if rscript is None:
        return "Rscript not on PATH (install R to run the R-parity suite)."
    if not _DRIVER.exists():
        return f"R driver missing at {_DRIVER}."
    if not _survey_available(rscript):
        return "R 'survey' package not installed (install.packages('survey'))."
    return None


#: Module-level skip: no R toolchain / no survey => the whole file is skipped,
#: never failed. Evaluated once at collection.
pytestmark = pytest.mark.skipif(
    _skip_reason() is not None,
    reason=_skip_reason() or "",
)


def _r_g_weights(
    calfun: str,
    matrix: np.ndarray,
    target: np.ndarray,
    initial_weights: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Run the R driver and return R ``survey``'s g-weights for a problem.

    Writes ``(A, b, w0)`` to CSVs, invokes ``scripts/survey_calibrate.R`` with
    the chosen ``calfun``, and reads back the g-weight column. Raises if the
    subprocess fails, so an R-side error is a loud test failure (once we have
    decided R is available), not a silent skip.

    Args:
        calfun: R ``survey`` calibration function: ``"raking"``, ``"linear"``,
            or ``"logit"``.
        matrix: The dense constraint matrix ``A`` (``n_targets x n_records``).
        target: The target vector ``b``.
        initial_weights: The design weights ``w0``.
        bounds: ``(L, U)`` g-weight bounds, required for ``calfun="logit"``.

    Returns:
        The length-``n_records`` g-weight vector R computed.
    """
    rscript = _rscript()
    assert rscript is not None  # guarded by the module skip
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        m_path = tmp_path / "matrix.csv"
        t_path = tmp_path / "target.csv"
        w_path = tmp_path / "weights.csv"
        g_path = tmp_path / "gweights.csv"
        np.savetxt(m_path, np.asarray(matrix, dtype=np.float64), delimiter=",")
        np.savetxt(t_path, np.asarray(target, dtype=np.float64))
        np.savetxt(w_path, np.asarray(initial_weights, dtype=np.float64))
        args = [
            rscript,
            str(_DRIVER),
            calfun,
            str(m_path),
            str(t_path),
            str(w_path),
            str(g_path),
        ]
        if bounds is not None:
            args += [str(bounds[0]), str(bounds[1])]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"R driver failed for calfun={calfun!r} "
                f"(exit {proc.returncode}):\n{proc.stderr}"
            )
        g = np.loadtxt(g_path)
        return np.atleast_1d(np.asarray(g, dtype=np.float64))


def _feasible_dense(seed: int, *, n_records: int = 60, n_targets: int = 4):
    """A small feasible dense problem (R's default epsilon can hit the targets)."""
    problem = feasible_problem(
        n_records=n_records, n_targets=n_targets, seed=seed
    ).problem
    return problem.dense(), problem.target, problem.initial_weights


def test_raking_matches_r_survey() -> None:
    """Our raking (F=exp) reproduces R ``survey``'s ``calfun="raking"``.

    Both minimize KL from the design weights subject to the calibration
    equations, so on a feasible surface they must land on the same g-weights.
    """
    a, b, w0 = _feasible_dense(seed=0)
    g_r = _r_g_weights("raking", a, b, w0)
    solution = raking_weights(a, b, w0)
    assert solution.converged
    np.testing.assert_allclose(
        solution.g_weights, g_r, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


def test_linear_greg_matches_r_survey() -> None:
    """Our GREG (F=1+u) reproduces R ``survey``'s ``calfun="linear"``.

    The Deville-Sarndal linear distance -- the generalized regression estimator.
    R's linear calibration is closed form; ours takes Newton steps to a tighter
    tolerance, so the two agree to R's convergence epsilon.
    """
    a, b, w0 = _feasible_dense(seed=1)
    g_r = _r_g_weights("linear", a, b, w0)
    solution = linear_weights(a, b, w0)
    assert solution.converged
    np.testing.assert_allclose(
        solution.g_weights, g_r, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


def test_logit_bounded_matches_r_survey() -> None:
    """Our logit calibration reproduces R ``survey``'s ``calfun="logit"``.

    The Deville-Sarndal logit calibration function confining g-weights to
    ``(L, U)``; this is exactly R ``survey``'s ``calfun="logit"`` with the same
    ``bounds``. On a feasible surface the interior solution matches R's.
    """
    a, b, w0 = _feasible_dense(seed=2)
    bounds = (0.2, 5.0)
    g_r = _r_g_weights("logit", a, b, w0, bounds=bounds)
    solution = logit_weights(a, b, w0, bounds=bounds)
    assert solution.converged
    np.testing.assert_allclose(
        solution.g_weights, g_r, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


def test_parity_holds_across_several_seeds() -> None:
    """Parity is not a single-draw coincidence: hold it over several problems.

    Runs raking (the workhorse) on a handful of independent feasible surfaces so
    a regression that only matched one particular matrix is caught.
    """
    for seed in range(4):
        a, b, w0 = _feasible_dense(seed=10 + seed, n_records=50, n_targets=3)
        g_r = _r_g_weights("raking", a, b, w0)
        solution = raking_weights(a, b, w0)
        assert solution.converged, seed
        np.testing.assert_allclose(
            solution.g_weights,
            g_r,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
            err_msg=f"raking parity failed at seed {seed}",
        )
