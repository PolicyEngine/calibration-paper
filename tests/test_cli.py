"""The ``cal`` CLI dispatches and its base-install commands run.

Base-install tests (no torch, no populace, no R): the console entry point exists,
the dispatcher routes and reports usage errors, and ``cal methods`` / ``cal demo``
run end to end and print the registry / diagnostics. These guard the entry point
declared in ``pyproject.toml`` (``cal = "calibration_paper.cli:main"``) -- a broken
entry point is a packaging bug that unit tests on the library alone would miss.
"""

from __future__ import annotations

import pytest

from calibration_paper.cli import main


def test_no_args_prints_usage_and_succeeds(capsys) -> None:
    """``cal`` with no command prints usage and exits 0."""
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage: cal <command>" in out
    assert "methods" in out and "demo" in out


def test_help_flag_prints_usage(capsys) -> None:
    """``cal --help`` prints usage and exits 0."""
    assert main(["--help"]) == 0
    assert "commands:" in capsys.readouterr().out


def test_unknown_command_is_a_usage_error(capsys) -> None:
    """An unknown command exits 2 and names the mistake on stderr."""
    code = main(["nope"])
    assert code == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


def test_methods_command_lists_every_registered_method(capsys) -> None:
    """``cal methods`` prints one line per registered method key."""
    from calibration_paper.methods import REGISTRY

    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    for key in REGISTRY:
        assert key in out


def test_methods_command_filters_by_family(capsys) -> None:
    """``cal methods --family gradient`` shows only the gradient methods."""
    assert main(["methods", "--family", "gradient"]) == 0
    out = capsys.readouterr().out
    assert "sgd" in out and "sgd_bounded" in out
    assert "raking" not in out


def test_demo_command_runs_and_reports_convergence(capsys) -> None:
    """``cal demo`` runs every classical method and exits 0 (all converge)."""
    code = main(["demo", "--seed", "0", "--n-records", "120"])
    assert code == 0  # feasible demo: every classical method converges
    out = capsys.readouterr().out
    assert "Classical calibration demo" in out
    # The table names each classical method.
    from calibration_paper.methods import MethodFamily, list_methods

    for key in list_methods(MethodFamily.CLASSICAL):
        assert key in out


def test_demo_command_is_deterministic_across_process_calls(capsys) -> None:
    """The same seed prints the same table (the demo is deterministic)."""
    main(["demo", "--seed", "5", "--n-records", "100"])
    first = capsys.readouterr().out
    main(["demo", "--seed", "5", "--n-records", "100"])
    second = capsys.readouterr().out
    assert first == second


@pytest.mark.parametrize("bad", [["methods", "--family", "nonsense"]])
def test_argparse_rejects_bad_options(bad) -> None:
    """A bad option to a subcommand is an argparse error (SystemExit)."""
    with pytest.raises(SystemExit):
        main(bad)
