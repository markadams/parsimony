#!/usr/bin/env python3
"""Mixed-precision search on Bratu Newton under the iteration-budget criterion.

Sweeps the budget slack to trace the Pareto curve of storage against iterations,
over real storage formats.  Time is applied afterwards from a cost model; it is
never the search objective (DESIGN.md, section 2).
"""
from __future__ import annotations
import sys, json
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from parsimony import DEFAULT_LADDER, IterationBudget, tune
from bratu import CASES, kernel

SLACKS = (0.0, 0.25, 1.0, 4.0)


def sweep(cases, label):
    print(f"\n== {label} ==")
    records = []
    header = None
    for slack in SLACKS:
        for pf in (False, True):
            r = tune(kernel, ladder=DEFAULT_LADDER, prefilter=pf, cases=cases,
                     criterion=IterationBudget(slack))
            if header is None:
                header = list(r.mixed)
                print(f"{'slack':>6} {'pre':>6} {'probes':>7} {'bits':>5} {'iters/ref':>9}  "
                      + " ".join(f"{v:>8}" for v in header))
            print(f"{slack:>6} {str(pf):>6} {r.total_probes:>7} {r.bits_after:>5} "
                  f"{r.mixed_error:>9.2f}  " + " ".join(f"{r.mixed[v]:>8}" for v in header))
            records.append({"label": label, "slack": slack, "prefilter": pf, **r.to_dict()})
    return records


out = sweep(CASES, "stored quantities only (F evaluated in fp64)")
out += sweep(tuple(replace(c, fine_residual=True) for c in CASES),
             "residual terms rounded before cancellation")
json.dump(out, open(Path(__file__).with_name("search.json"), "w"), indent=1)
