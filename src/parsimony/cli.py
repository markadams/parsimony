"""Command line interface.  Emits JSON by default so an agent can read a verdict
without the numbers passing through a transcript.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import DEFAULT_LADDER, TuneReport, tune
from .formats import Format, virtual


def _load(spec: str):
    """Import ``module:function`` or ``path/to/file.py:function``."""
    if ":" not in spec:
        raise SystemExit(f"expected module:function, got {spec!r}")
    mod, _, func = spec.partition(":")
    if mod.endswith(".py"):
        path = Path(mod).resolve()
        sys.path.insert(0, str(path.parent))
        mod = path.stem
    module = importlib.import_module(mod)
    try:
        return getattr(module, func)
    except AttributeError:
        raise SystemExit(f"{mod} has no attribute {func!r}") from None


def _ladder(spec: str | None) -> tuple[Format, ...]:
    if not spec:
        return DEFAULT_LADDER
    by_name = {f.name: f for f in DEFAULT_LADDER}
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token in by_name:
            out.append(by_name[token])
        elif token.startswith("v") and token[1:].isdigit():
            out.append(virtual(int(token[1:])))
        else:
            raise SystemExit(f"unknown format {token!r}")
    return tuple(out)


def _text(report: TuneReport) -> str:
    lines = [
        f"variables      {report.variables}",
        f"tolerance      {report.tolerance:g}",
        f"uniform floor  {report.uniform_floor or 'none'}  ({report.uniform_probes} probes)",
        f"mixed          {report.mixed_counts}  ({report.mixed_probes} probes)",
        f"mixed error    {report.mixed_error:.3e}",
        f"storage        {report.bits_before} -> {report.bits_after} bits"
        f"  ({report.bit_reduction:.1%} smaller)",
    ]
    if report.trace:
        lines.append("per level      " + ", ".join(
            f"{name}: {got}/{tried}" for name, got, tried in report.trace))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parsimony")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tune", help="find the lowest precision meeting a tolerance")
    t.add_argument("kernel", help="module:function or path/to/file.py:function")
    t.add_argument("--tol", type=float, default=1e-6,
                   help="relative error tolerance (default: 1e-6)")
    t.add_argument("--ladder", default=None,
                   help="comma-separated formats, high to low, e.g. fp64,fp32,bf16 "
                        "or virtual widths like v53,v40,v24")
    t.add_argument("--text", action="store_true", help="human-readable instead of JSON")

    args = parser.parse_args(argv)
    report = tune(_load(args.kernel), tol=args.tol, ladder=_ladder(args.ladder))
    print(_text(report) if args.text else json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
