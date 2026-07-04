"""The registry imports and lists on the base install, with no heavy-dep leak.

These are the "the surface is well-formed" tests: they must pass on the base
install (no torch, no populace), so they guard the lazy-import discipline that
lets CI run without the ``methods`` extra.
"""

from __future__ import annotations

import subprocess
import sys

from calibration_paper import methods
from calibration_paper.methods import (
    CLASSICAL_KEYS,
    GRADIENT_KEYS,
    REGISTRY,
    MethodFamily,
    get_method,
    list_methods,
)


def test_registry_has_the_seven_plan_methods_plus_bounded_chi_square() -> None:
    """PLAN.md's seven keys are all registered (chi_square split into two)."""
    # PLAN.md lists raking, greg, chi_square, entropy, logit_bounded, sgd,
    # sgd_bounded. We additionally register chi_square_bounded so the
    # "chi-square with and without bounds" note in PLAN.md is two runnable keys.
    for key in (
        "raking",
        "greg",
        "chi_square",
        "chi_square_bounded",
        "entropy",
        "logit_bounded",
        "sgd",
        "sgd_bounded",
    ):
        assert key in REGISTRY, key


def test_families_partition_the_registry() -> None:
    """Every method is classical or gradient; the groupings are exhaustive."""
    assert set(CLASSICAL_KEYS) | set(GRADIENT_KEYS) == set(REGISTRY)
    assert set(CLASSICAL_KEYS) & set(GRADIENT_KEYS) == set()
    assert set(GRADIENT_KEYS) == {"sgd", "sgd_bounded"}


def test_list_methods_filters_by_family() -> None:
    """``list_methods(family)`` returns exactly that family's keys."""
    assert list_methods(MethodFamily.CLASSICAL) == list(CLASSICAL_KEYS)
    assert list_methods(MethodFamily.GRADIENT) == list(GRADIENT_KEYS)
    assert list_methods() == list(REGISTRY)


def test_get_method_raises_helpfully_on_unknown_key() -> None:
    """An unknown key names the registered methods in the error."""
    try:
        get_method("does_not_exist")
    except KeyError as exc:
        assert "raking" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("get_method should raise on an unknown key.")


def test_every_method_has_citation_and_description() -> None:
    """Each method is grounded: a nonempty description and BibTeX citation key."""
    for key, method in REGISTRY.items():
        assert method.description.strip(), key
        assert method.citation_key.strip(), key


def test_importing_registry_does_not_import_torch_or_populace() -> None:
    """Lazy imports hold: the registry module pulls in no heavy method package.

    This is the load-bearing invariant -- if a constructor imported torch or
    populace at module scope, CI (base install) would fail to import the
    registry at all.

    Checked in a *fresh* subprocess, not against this process's ``sys.modules``:
    with the ``methods`` extra installed, other tests in the same session import
    torch/populace, so the ambient module table is polluted and cannot answer
    "did importing the registry import torch". A clean interpreter that imports
    only the registry and inspects its own ``sys.modules`` is the honest check,
    and it holds whether or not the heavy deps are installed.
    """
    probe = (
        "import sys;"
        "import calibration_paper.methods;"
        "leaked = [m for m in ('torch', 'populace', 'populace.calibrate')"
        " if m in sys.modules];"
        "print(','.join(leaked));"
        "sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "importing calibration_paper.methods dragged in a heavy method package: "
        f"{result.stdout.strip()!r} (the constructors must import lazily)."
    )


def test_bounded_methods_declare_their_bounds() -> None:
    """The bounded classical methods carry the g-weight bounds they enforce."""
    assert methods.get_method("chi_square_bounded").respects_bounds == (0.2, 5.0)
    assert methods.get_method("logit_bounded").respects_bounds == (0.2, 5.0)
    assert methods.get_method("raking").respects_bounds is None
    assert methods.get_method("greg").respects_bounds is None


def test_positivity_guarantees_are_declared_correctly() -> None:
    """Only GREG and unbounded chi-square may go negative; the rest are positive."""
    positive = {k for k, m in REGISTRY.items() if m.guarantees_positive}
    may_be_negative = set(REGISTRY) - positive
    assert may_be_negative == {"greg", "chi_square"}
