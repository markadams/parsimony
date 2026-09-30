"""The HDCG evaluator's plumbing, checked against a stand-in for the driver.

Grid is not available here and a real solve costs minutes on four GPUs, so what
is pinned is everything between parsimony and the binary: the command line the
driver will have to parse, the JSON contract it has to emit, the memoization
that keeps a search affordable, and the no-op guard that stops a mis-attached
rounding hook from being reported as a free variable.

The stand-in has a knee per variable, so the search has something real to find:
subspace tolerates 16 bits, stencil 30, evec 8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HDCG = Path(__file__).resolve().parents[1] / "casestudies" / "hdcg"
sys.path.insert(0, str(HDCG))

from grid_eval import (  # noqa: E402
    LADDER, VARIABLES, GridHDCGEvaluator, RoundingNoOp, run,
)
from parsimony.criteria import IterationBudget  # noqa: E402
from parsimony.formats import virtual  # noqa: E402

# Bits at or above which each variable is free; below, iterations climb fast.
KNEES = {"subspace": 16, "stencil": 30, "evec": 8}

FAKE = '''
import argparse, json
ap = argparse.ArgumentParser()
for v in ("subspace", "stencil", "evec"):
    ap.add_argument("--round-" + v, default="v53")
ap.add_argument("--case", type=int, default=0)
ap.add_argument("--json", required=True)
ap.add_argument("--grid"); ap.add_argument("--mpi")
a = ap.parse_args()

KNEES = {knees!r}
NOOP = {noop!r}
bits = {{v: 53 if getattr(a, "round_" + v) == "v53"
         else int(getattr(a, "round_" + v).lstrip("v")) for v in KNEES}}

iters, delta = 20 + a.case, {{}}
for v, knee in KNEES.items():
    if bits[v] < 53:
        delta[v] = 0.0 if v in NOOP else 2.0 ** -bits[v]
    if bits[v] < knee:
        iters += 4 * (knee - bits[v])
json.dump({{"converged": iters < 200, "iterations": iters,
           "true_resid": 1e-9, "t_setup": 1.0, "t_solve": 0.1 * iters,
           "delta": delta, "case": a.case}}, open(a.json, "w"))
'''


def make_evaluator(tmp_path, noop=(), knees=None, **kw):
    driver = tmp_path / "fake_driver.py"
    driver.write_text(FAKE.format(knees=dict(knees or KNEES), noop=list(noop)))
    return GridHDCGEvaluator(
        binary=str(driver), launcher=[sys.executable],
        grid_args=["--grid", "16.16.16.32", "--mpi", "1.1.1.4"],
        cases=(0, 1, 2), workdir=tmp_path, **kw,
    )


def cfg(**kw):
    """A configuration with rounding off except where named."""
    c = {v: LADDER[0] for v in VARIABLES}
    c.update({k: virtual(b) for k, b in kw.items()})
    return c


def test_reference_run_defines_the_budget(tmp_path):
    ev = make_evaluator(tmp_path)
    space = ev.discover()
    assert list(space.variables) == list(VARIABLES)

    r = ev.evaluate(cfg())
    assert r.passed and r.error == pytest.approx(1.0)


def test_command_line_matches_the_driver_contract(tmp_path):
    ev = make_evaluator(tmp_path)
    cmd = ev._command(cfg(subspace=16), case=2, json_path=tmp_path / "o.json")
    assert cmd[0] == sys.executable
    for flag in ("--round-subspace", "--round-stencil", "--round-evec",
                 "--case", "--json"):
        assert flag in cmd
    assert cmd[cmd.index("--round-subspace") + 1] == "v16"
    assert cmd[cmd.index("--round-evec") + 1] == "v53"


def test_above_the_knee_is_free_below_it_is_not(tmp_path):
    ev = make_evaluator(tmp_path)
    ev.discover()
    assert ev.evaluate(cfg(subspace=16)).passed
    assert not ev.evaluate(cfg(subspace=11)).passed
    assert not ev.evaluate(cfg(stencil=24)).passed
    assert ev.evaluate(cfg(evec=8)).passed


def test_probes_are_memoized(tmp_path):
    ev = make_evaluator(tmp_path)
    ev.discover()
    ev.evaluate(cfg(subspace=24))
    solves = ev.solves
    for _ in range(5):
        ev.evaluate(cfg(subspace=24))
    assert ev.solves == solves
    assert ev.probes > solves


def test_cache_survives_a_new_evaluator(tmp_path):
    """An allocation that dies mid-search must not re-solve what it already did."""
    first = make_evaluator(tmp_path)
    first.discover()
    first.evaluate(cfg(stencil=30))

    second = make_evaluator(tmp_path)
    second.discover()
    second.evaluate(cfg(stencil=30))
    assert second.solves == 0


def test_a_hook_that_rounds_nothing_stops_the_search(tmp_path):
    """The failure that would otherwise read as 'this input needs no bits'."""
    ev = make_evaluator(tmp_path, noop=["stencil"])
    ev.discover()
    ev.evaluate(cfg(subspace=24))  # unaffected variable still works

    with pytest.raises(RoundingNoOp, match="stencil"):
        ev.evaluate(cfg(stencil=24))


def test_missing_json_is_an_error_not_a_divergence(tmp_path):
    """A driver that crashes must not be scored as 'this format diverged'."""
    driver = tmp_path / "fake_driver.py"
    driver.write_text("import sys; sys.exit(1)")
    ev = GridHDCGEvaluator(binary=str(driver), launcher=[sys.executable],
                           cases=(0,), workdir=tmp_path)
    with pytest.raises(RuntimeError, match="no JSON"):
        ev.discover()


def test_slack_sweep_costs_no_extra_solves(tmp_path):
    ev = make_evaluator(tmp_path)
    ev.discover()
    ev.evaluate(cfg(subspace=11))
    solves = ev.solves

    assert not ev.evaluate(cfg(subspace=11)).passed
    ev.recalibrate(IterationBudget(slack=4.0))
    assert ev.evaluate(cfg(subspace=11)).passed
    assert ev.solves == solves


def test_search_recovers_the_knees(tmp_path):
    ev = make_evaluator(tmp_path)
    report = run(ev, slacks=[0.0])
    found = report["results"][0]

    # The prefilter floors are the knees, to the resolution of the ladder.
    for v, knee in KNEES.items():
        assert int(found["floors"][v].lstrip("v")) >= knee, found["floors"]
    assert found["passed"]
    assert found["bits_after"] < found["bits_before"]
    # The plan budgets 40-80 solves for the whole study; one slack over three
    # variables and seven rungs must stay a small fraction of that.
    assert report["solves"] <= 25, report["solves"]


def test_extra_slacks_are_nearly_free(tmp_path):
    """The Pareto sweep reuses solves; only new configurations cost GPU time."""
    ev = make_evaluator(tmp_path)
    one = run(ev, slacks=[0.0])["solves"]

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    three = run(make_evaluator(fresh), slacks=[0.0, 0.1, 0.25])["solves"]
    assert three < 2 * one, (one, three)


def test_probe_log_records_every_run(tmp_path):
    ev = make_evaluator(tmp_path)
    ev.discover()
    ev.evaluate(cfg(evec=16))
    lines = [json.loads(l) for l in (tmp_path / "probes.jsonl").read_text().splitlines()]
    assert len(lines) == 2 * len(ev.cases)  # reference + one probe, three cases
    assert all("cmd" in l and "returncode" in l for l in lines)
