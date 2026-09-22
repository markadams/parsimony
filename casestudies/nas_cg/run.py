#!/usr/bin/env python3
"""Reproduce the conjugate-gradient case study.  Writes results.json.

    python casestudies/nas_cg/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from nas_cg import CASES, kernel  # noqa: E402
from parsimony import tune  # noqa: E402

TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10)
ORDER = ("A", "z", "r", "q", "p", "rho", "rho0", "d", "alpha", "beta")


def main() -> int:
    for i, (A, _, niter) in enumerate(CASES):
        print(f"case{i}: n = {A.shape[0]:>4}, {niter} iterations")

    header = f"\n{'tol':>7} {'pre':>6} {'probes':>7} {'bits':>5}  "
    print(header + " ".join(f"{v:>8}" for v in ORDER))

    records = []
    for tol in TOLERANCES:
        for prefilter in (False, True):
            r = tune(kernel, tol=tol, prefilter=prefilter, cases=CASES)
            records.append({"tol": tol, "prefilter": prefilter, **r.to_dict()})
            print(f"{tol:>7.0e} {str(prefilter):>6} {r.total_probes:>7} {r.bits_after:>5}  "
                  + " ".join(f"{r.mixed[v]:>8}" for v in ORDER))

    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(records, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
