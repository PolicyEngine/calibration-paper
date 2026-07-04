"""``cal methods``: list the calibrator surface.

Prints every registered method -- key, family, positivity guarantee, declared
bounds, and citation key -- straight from :data:`calibration_paper.methods.REGISTRY`,
so the paper's "what is compared against what" is inspectable from the command
line without reading the source. Base install only: the registry imports without
torch or populace.
"""

from __future__ import annotations

import argparse

from calibration_paper.methods import REGISTRY, MethodFamily


def main(argv: list[str] | None = None) -> int:
    """Print the registered methods, optionally filtered to one family.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]`` via argparse.

    Returns:
        ``0``.
    """
    parser = argparse.ArgumentParser(
        prog="cal methods",
        description="List the calibrator surface (the registered methods).",
    )
    parser.add_argument(
        "--family",
        choices=(MethodFamily.CLASSICAL, MethodFamily.GRADIENT),
        default=None,
        help="Only show methods in this family (default: all).",
    )
    args = parser.parse_args(argv)

    keys = [
        key
        for key, method in REGISTRY.items()
        if args.family is None or method.family == args.family
    ]
    key_width = max(len(k) for k in keys)
    family_width = max(len(REGISTRY[k].family) for k in keys)

    header = f"{'key':<{key_width}}  {'family':<{family_width}}  pos  bounds"
    print(header)
    print("-" * len(header))
    for key in keys:
        method = REGISTRY[key]
        positive = "yes" if method.guarantees_positive else "no "
        bounds = (
            f"[{method.respects_bounds[0]:g}, {method.respects_bounds[1]:g}]"
            if method.respects_bounds is not None
            else "-"
        )
        print(
            f"{key:<{key_width}}  {method.family:<{family_width}}  "
            f"{positive}  {bounds}"
        )
        print(f"{'':<{key_width}}    {method.description}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module-run convenience
    raise SystemExit(main())
