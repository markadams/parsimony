"""Acceptance criteria are the part of the tool that decides what 'passes' means,
so their semantics are pinned exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parsimony.criteria import IterationBudget, RelativeError, by_case


def test_by_case_groups_prefixed_keys_and_defaults_unprefixed_to_zero():
    g = by_case({"case0.x": 1, "case1.x": 2, "case1.y": 3, "z": 4})
    assert g == {0: {"x": 1, "z": 4}, 1: {"x": 2, "y": 3}}


def test_relative_error_matches_the_old_evaluator_semantics():
    c = RelativeError(1e-6)
    c.calibrate({"a": np.array([1.0, 2.0]), "b": 0.0})
    ok, err, _ = c.accept({"a": np.array([1.0, 2.0 + 1e-7]), "b": 0.0})
    assert ok and err == pytest.approx(5e-8)
    ok, err, _ = c.accept({"a": np.array([1.0, 2.0]), "b": 1e-7})  # zero ref: absolute
    assert ok and err == pytest.approx(1e-7)
    ok, err, _ = c.accept({"a": np.array([np.nan, 2.0]), "b": 0.0})
    assert not ok and math.isinf(err)
    ok, err, _ = c.accept({"a": np.array([1.0, 2.0])})  # missing output
    assert not ok and math.isinf(err)


def test_iteration_budget_zero_slack_admits_exactly_the_reference_count():
    c = IterationBudget(slack=0.0)
    c.calibrate({"case0.converged": True, "case0.iterations": 5,
                 "case1.converged": True, "case1.iterations": 7})
    ok, score, meta = c.accept({"case0.converged": True, "case0.iterations": 5,
                                "case1.converged": True, "case1.iterations": 7})
    assert ok and score == 1.0 and meta["iterations"] == {0: 5, 1: 7}
    ok, score, _ = c.accept({"case0.converged": True, "case0.iterations": 6,
                             "case1.converged": True, "case1.iterations": 7})
    assert not ok and score == pytest.approx(1.2)


def test_iteration_budget_slack_is_applied_per_case_with_ceil():
    c = IterationBudget(slack=0.25)
    c.calibrate({"case0.converged": True, "case0.iterations": 5,
                 "case1.converged": True, "case1.iterations": 3})
    # ceil(1.25*5)=7 and ceil(1.25*3)=4 admitted; 8 or 5 not.
    assert c.accept({"case0.converged": True, "case0.iterations": 7,
                     "case1.converged": True, "case1.iterations": 4})[0]
    assert not c.accept({"case0.converged": True, "case0.iterations": 8,
                         "case1.converged": True, "case1.iterations": 4})[0]
    assert not c.accept({"case0.converged": True, "case0.iterations": 7,
                         "case1.converged": True, "case1.iterations": 5})[0]


def test_iteration_budget_divergence_is_infinite_score():
    c = IterationBudget(slack=10.0)
    c.calibrate({"converged": True, "iterations": 4})
    ok, score, meta = c.accept({"converged": False, "iterations": 60})
    assert not ok and math.isinf(score) and meta == {"diverged_case": 0}


def test_iteration_budget_refuses_an_unconverged_reference():
    with pytest.raises(ValueError):
        IterationBudget().calibrate({"converged": False, "iterations": 60})
