"""End-to-end check on the worked example from Blame Analysis, Figure 2.

The double-precision answer reproduces the paper's 0.5000000113 exactly.  For
the all-single answer the test pins to hardware float32 rather than to the
paper's printed constant: the emulator and numpy agree on 0.4999980926513672,
and the paper's figure appears to read 0.4999980927 (the trailing digits match;
a transcription of the sixth decimal is the only difference).  Hardware is the
stronger oracle anyway -- it catches an emulator bug, which a copied constant
cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from parsimony import FP32, FP64, Goal, NumpyEvaluator, tune, uniform  # noqa: E402
from mpow import kernel  # noqa: E402

PAPER_DOUBLE = 0.5000000113


def _native(dtype):
    """Run the same kernel with a real numpy dtype instead of the emulator."""
    return float(kernel(lambda _name, value: dtype(value))["res"])


def test_double_matches_paper():
    ev = NumpyEvaluator(kernel)
    ev.discover()
    assert ev._reference["res"] == pytest.approx(PAPER_DOUBLE, abs=5e-10)


def test_all_single_matches_hardware():
    ev = NumpyEvaluator(kernel, Goal(rel_tol=1e-12))
    space = ev.discover()
    ev.evaluate(uniform(space, FP32))
    emulated = kernel(_recorder(uniform(space, FP32)))["res"]
    assert emulated == _native(np.float32)


def _recorder(config):
    from parsimony import Precision

    return Precision(config)


def test_cancellation_costs_about_two_digits():
    """Terms near 25 summing to 0.5 -- the loss is real, not an artifact."""
    ev = NumpyEvaluator(kernel, Goal(rel_tol=1e-12))
    space = ev.discover()
    res = ev.evaluate(uniform(space, FP32))
    assert 1e-6 < res.error < 1e-5


def test_cancellation_blocks_uniform_fp32_at_tight_tolerance():
    report = tune(kernel, tol=1e-8)
    assert report.uniform_floor == FP64.name


def test_loose_tolerance_admits_fp32():
    report = tune(kernel, tol=1e-4)
    assert report.uniform_floor is not None
    assert report.uniform_floor != FP64.name


def test_mixed_search_beats_uniform_at_tight_tolerance():
    """Where uniform fp32 fails, ddmin should still lower *something*."""
    report = tune(kernel, tol=1e-8)
    assert report.uniform_floor == FP64.name
    assert report.bits_after < report.bits_before, report.mixed


def test_factor_is_lowerable():
    """The paper's finding: mpow's `factor` holds small exact integers."""
    report = tune(kernel, tol=1e-8)
    assert report.mixed["mpow.factor"] != FP64.name, report.mixed


def test_tighter_tolerance_never_lowers_more():
    loose = tune(kernel, tol=1e-4)
    tight = tune(kernel, tol=1e-10)
    assert tight.bits_after >= loose.bits_after


def test_report_is_json_serializable():
    import json

    json.dumps(tune(kernel, tol=1e-6).to_dict())
