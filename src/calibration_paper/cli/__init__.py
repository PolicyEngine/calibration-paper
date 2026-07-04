"""The ``cal`` command-line interface.

A small dispatcher, mirroring the sibling papers' CLIs (``l0`` in sparsity-paper,
``imp`` in imputation-paper): ``cal <command> [args...]`` routes to the matching
driver's ``main()``. Commands are imported lazily so a command that needs a heavy
dependency does not drag it in for a command that does not -- the method surface
(``cal methods``) and the classical demo (``cal demo``) run on the base install
with no torch, no populace, no R.

The console entry point is declared in ``pyproject.toml`` as
``cal = "calibration_paper.cli:main"``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

#: command name -> (module, one-line help). Each module must expose ``main()``.
_COMMANDS: dict[str, tuple[str, str]] = {
    "methods": (
        "calibration_paper.cli.methods",
        "List the calibrator surface (every registered method).",
    ),
    "demo": (
        "calibration_paper.cli.demo",
        "Run every classical method on a synthetic problem and print diagnostics.",
    ),
}


def _usage() -> str:
    """The top-level usage string listing every command."""
    width = max(len(name) for name in _COMMANDS)
    lines = ["usage: cal <command> [args...]", "", "commands:"]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in _COMMANDS.items()]
    lines += ["", "Run 'cal <command> --help' for a command's own options."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``cal <command> [args...]`` to the matching driver.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The command's exit status (``0`` on success, ``2`` for a usage error).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        print(f"cal: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    module = importlib.import_module(_COMMANDS[command][0])
    # Each driver parses sys.argv; present it the subcommand's own args.
    saved = sys.argv
    sys.argv = [f"cal {command}", *rest]
    try:
        result = module.main()
    finally:
        sys.argv = saved
    return result if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover - module-run convenience
    raise SystemExit(main())
