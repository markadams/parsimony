"""The prefilter's cost/quality trade, and the input-count failure it exposed.

Two behaviours are pinned here because both were discovered rather than designed,
and a regression in either would be silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from parsimony import (  # noqa: E402
    FP32, FP64, Goal, NumpyEvaluator, SoloSensitivity, tune,
)
from parsimony.formats import DEFAULT_LADDER, rank  # noqa: E402
from mpow import CASES, kernel  # noqa: E402


# --- The single-input failure -------------------------------------------------
#
# At a = 1.84089642 the result is within 2.3e-8 of exactly 0.5, which an 8-bit
# format represents exactly.  Given only that input, the search can meet a loose
# tolerance by coarsening the *output* until it quantizes onto that point: a
# downstream format absorbing upstream error, not an accurate computation.

def _below_fp32(report):
    return [v for v, name in report.mixed.items()
            if rank({f.name: f for f in DEFAULT_LADDER}[name]) < rank(FP32)]


def test_single_input_admits_a_degenerate_configuration():
    """With one input, even the *input* variable drops below fp32."""
    report = tune(kernel, tol=1e-4, prefilter=False)
    assert "a" in _below_fp32(report), report.mixed


def test_multiple_inputs_remove_it():
    """With five inputs, no single lattice point is lucky for all of them."""
    report = tune(kernel, tol=1e-4, prefilter=False, cases=CASES)
    assert "a" not in _below_fp32(report), report.mixed


def test_multiple_inputs_restore_tolerance_monotonicity():
    """Tightening the tolerance must not admit more lowering."""
    bits = [tune(kernel, tol=t, prefilter=False, cases=CASES).bits_after
            for t in (1e-4, 1e-6, 1e-8)]
    assert bits == sorted(bits), bits


# --- The prefilter's trade ----------------------------------------------------

@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8])
def test_prefilter_costs_far_fewer_evaluations(tol):
    without = tune(kernel, tol=tol, prefilter=False, cases=CASES)
    with_ = tune(kernel, tol=tol, prefilter=True, cases=CASES)
    assert with_.total_probes * 4 < without.total_probes, (
        with_.total_probes, without.total_probes)


@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8])
def test_prefilter_costs_at_most_a_little_quality(tol):
    """It restricts the space, so it may find a wider assignment -- but not much."""
    without = tune(kernel, tol=tol, prefilter=False, cases=CASES)
    with_ = tune(kernel, tol=tol, prefilter=True, cases=CASES)
    assert with_.bits_after <= without.bits_after * 1.25, (
        with_.bits_after, without.bits_after)


def test_prefilter_shrinks_the_space():
    report = tune(kernel, tol=1e-8, prefilter=True, cases=CASES)
    assert report.space_after < report.space_before / 1000


def test_prefilter_agrees_at_tight_tolerance():
    """Where the answer is forced, the cheap and expensive routes must concur."""
    without = tune(kernel, tol=1e-8, prefilter=False, cases=CASES)
    with_ = tune(kernel, tol=1e-8, prefilter=True, cases=CASES)
    assert with_.bits_after == without.bits_after


# --- Soundness of the floors --------------------------------------------------

def test_floors_are_consistent_with_a_solo_scan():
    """The bisection must agree with exhaustively scanning the ladder."""
    ev = NumpyEvaluator(kernel, Goal(rel_tol=1e-4), cases=CASES)
    space = ev.discover()
    pre = SoloSensitivity()
    narrowed = pre.analyze(space, ev.evaluate)

    for v in space.variables:
        lowest = None
        for fmt in DEFAULT_LADDER:
            cfg = {x: FP64 for x in space.variables}
            cfg[v] = fmt
            if ev.evaluate(cfg).passed:
                lowest = fmt
        assert narrowed.floors[v] == lowest, (v, narrowed.floors[v], lowest)


def test_unlowerable_variables_are_pinned():
    ev = NumpyEvaluator(kernel, Goal(rel_tol=1e-10), cases=CASES)
    space = ev.discover()
    narrowed = SoloSensitivity().analyze(space, ev.evaluate)
    for v, fmt in narrowed.pinned.items():
        assert fmt == FP64
        assert narrowed.floors[v] == FP64
