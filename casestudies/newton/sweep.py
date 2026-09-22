#!/usr/bin/env python3
"""One variable at a time down a virtual-width ladder: iterations to converge.

Checks the DESIGN.md predictions: which quantities are rate-limiting (stored
Jacobian, stored step, stored residual) and which are accuracy-limiting (stored
iterate, residual terms rounded before they cancel), and the mechanism behind
each threshold.  Writes sweep.json.
"""
from __future__ import annotations
import sys, math, json
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import numpy as np
from parsimony import Precision, virtual
from bratu import CASES, newton, condition_number

BITS = (53, 46, 44, 42, 40, 38, 36, 34, 32, 28, 24, 20, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4)


def run(case, cfg):
    """Iterations if it truly converged; None if it did not; 'F' if the solver's
    own stopping test fired but the fp64 residual says otherwise (false stop)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = newton(Precision(cfg), case)
    if out["converged"]:
        return out["iterations"]
    return "F" if out["false_stop"] else None


def cell(v):
    return "-" if v is None else str(v)


def table(title, cfg_for_p, cases):
    print(f"\n{title}   (- = did not converge, F = false stop)")
    print(f"{'p':>4} " + " ".join(f"{'case'+str(i):>7}" for i in range(len(cases))))
    rows = {}
    for p in BITS:
        cells = [run(c, cfg_for_p(p)) for c in cases]
        rows[p] = cells
        print(f"{p:>4} " + " ".join(f"{cell(n):>7}" for n in cells))
    return rows


kappas = [condition_number(c) for c in CASES]
nref = [run(c, {}) for c in CASES]
print("case  k  lam    kappa  log2k  n_ref | predicted bits:  u ~ log2k+log2(1/rtol)"
      "   F-terms ~ log2(1/rtol)+3   J-knee ~ log2((4/h^2)/lam)")
for i, (c, kap, n) in enumerate(zip(CASES, kappas, nref)):
    lr = math.log2(1 / c.rtol)
    h2 = (c.k + 1) ** 2
    print(f"{i:>4} {c.k:>2} {c.lam:>4} {kap:8.1f} {math.log2(kap):6.2f} {n:>5} |"
          f"  u~{math.log2(kap) + lr:5.1f}   F-terms~{lr + 3:5.1f}"
          f"   J~{math.log2(4 * h2 / c.lam):5.1f}")

results = {"kappa": kappas, "n_ref": nref, "bits": list(BITS)}
results["J"] = table("Stored Jacobian J at p bits", lambda p: {"J": virtual(p)}, CASES)
results["delta"] = table("Stored Newton step delta at p bits", lambda p: {"delta": virtual(p)}, CASES)
results["F"] = table("Stored residual F (evaluated in fp64) at p bits", lambda p: {"F": virtual(p)}, CASES)
results["u"] = table("Stored iterate u at p bits", lambda p: {"u": virtual(p)}, CASES)
FINE = tuple(replace(c, fine_residual=True) for c in CASES)
results["F_terms"] = table("Residual terms F.lap, F.exp rounded BEFORE cancellation, at p bits",
                           lambda p: {"F.lap": virtual(p), "F.exp": virtual(p)}, FINE)
json.dump(results, open(Path(__file__).with_name("sweep.json"), "w"), indent=1)
