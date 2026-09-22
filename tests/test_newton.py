"""Bratu Newton: the design's predictions, pinned as structure not as numbers.

DESIGN.md predicted, before running anything, which quantities are rate-limiting
and which are accuracy-limiting, and what the mechanism is for each.  These tests
hold the tool to that: if a refactor makes the stored Jacobian look expensive or
the stored iterate look cheap, something is wrong with the tool, not the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from parsimony import FP32, FP64, IterationBudget, NumpyEvaluator, Precision, virtual  # noqa: E402
from bratu import CASES, Case, newton  # noqa: E402


def iters(case, cfg):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = newton(Precision(cfg), case)
    return out


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"k{c.k}-lam{c.lam}")
def test_reference_converges_quadratically_fast(case):
    out = iters(case, {})
    assert out["converged"] and out["iterations"] <= 6
    assert not out["false_stop"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"k{c.k}-lam{c.lam}")
def test_stored_jacobian_is_rate_limiting_not_fatal(case):
    """Low-precision J degrades to Picard iteration: slower, never divergent."""
    ref = iters(case, {})["iterations"]
    assert iters(case, {"J": virtual(16)})["iterations"] == ref
    low = iters(case, {"J": virtual(6)})
    assert low["converged"] and low["iterations"] > ref


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"k{c.k}-lam{c.lam}")
def test_stored_residual_is_nearly_free(case):
    """A rounded stored F perturbs the right-hand side relatively: no kappa."""
    ref = iters(case, {})["iterations"]
    low = iters(case, {"F": virtual(6)})
    assert low["converged"] and low["iterations"] <= ref + 2


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"k{c.k}-lam{c.lam}")
def test_stored_iterate_is_accuracy_limiting(case):
    """u cannot be stored in fp32 and meet a 1e-10 residual: floor ~ u_p * kappa."""
    assert not iters(case, {"u": FP32})["converged"]
    assert iters(case, {"u": virtual(44)})["converged"]


def test_rounded_residual_terms_produce_false_convergence_somewhere():
    """The solver's own test fires on an exactly-zero rounded residual."""
    fine = Case(k=16, lam=6.0, fine_residual=True)
    seen = [iters(fine, {"F.lap": virtual(p), "F.exp": virtual(p)})["false_stop"]
            for p in (10, 8, 6, 5, 4)]
    assert any(seen)


def test_evaluated_residual_needs_far_more_bits_than_stored_residual():
    """The storage-vs-arithmetic distinction, as a number."""
    case = Case(k=16, lam=6.0, fine_residual=True)
    stored_ok = iters(Case(k=16, lam=6.0), {"F": virtual(8)})["converged"]
    terms_ok = iters(case, {"F.lap": virtual(8), "F.exp": virtual(8)})["converged"]
    assert stored_ok and not terms_ok


def test_iteration_budget_criterion_composes_with_the_evaluator():
    ev = NumpyEvaluator(newton, cases=CASES, criterion=IterationBudget(slack=0.0))
    space = ev.discover()
    assert set(space.variables) == {"u", "F", "J", "delta"}
    assert ev.evaluate({v: FP64 for v in space.variables}).passed
    r = ev.evaluate({"J": virtual(6), "u": FP64, "F": FP64, "delta": FP64})
    assert not r.passed and r.error > 1.0  # converged, but over budget
