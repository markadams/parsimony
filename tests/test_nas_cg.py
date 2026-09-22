"""The conjugate-gradient case study, and the design errors it nearly shipped.

The structural findings are pinned rather than the exact formats, because the
formats depend on ddmin's path and would make this a change-detector.  What must
not regress is the *character* of the answer: scalars cheaper than vectors,
storage monotone in tolerance, and an operator whose entries are not free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from parsimony import FP32, FP64, FP8_E4M3, Goal, NumpyEvaluator, tune  # noqa: E402
from parsimony.formats import DEFAULT_LADDER, rank, round_to  # noqa: E402
from nas_cg import CASES, diffusion_2d, kernel  # noqa: E402

BY_NAME = {f.name: f for f in DEFAULT_LADDER}


def fmt(report, variable):
    return BY_NAME[report.mixed[variable]]


@pytest.fixture(scope="module")
def reports():
    return {tol: tune(kernel, tol=tol, prefilter=False, cases=CASES)
            for tol in (1e-4, 1e-6, 1e-8, 1e-10)}


# --- The operator must not be free -------------------------------------------

def test_operator_entries_are_not_exactly_representable():
    """A constant-coefficient Laplacian has entries 4 and -1, exact in fp8.

    That made `A` appear tunable to 8 bits at every tolerance -- an artifact of
    rounding a no-op, not a finding about storage.
    """
    for A, _, _ in CASES:
        entries = A[A != 0]
        assert not np.any(round_to(entries, FP8_E4M3) == entries)


def test_cases_span_a_range_of_conditioning():
    conds = [np.linalg.cond(A) for A, _, _ in CASES]
    assert conds == sorted(conds)
    assert conds[-1] / conds[0] > 4


# --- Iteration counts must be calibrated, not fixed ---------------------------

def test_iteration_counts_are_calibrated_per_case():
    counts = [niter for _, _, niter in CASES]
    assert counts == sorted(counts), counts
    assert len(set(counts)) == len(counts), "a shared count would idle the easy cases"


def test_reference_converges_without_entering_the_noise_floor():
    """Converged enough to leave headroom at 1e-10, not so far that rho underflows."""
    ev = NumpyEvaluator(kernel, Goal(rel_tol=1e-8), cases=CASES)
    ev.discover()
    for z in ev._reference.values():
        err = float(np.max(np.abs(z - 1.0)))
        assert 1e-15 < err < 1e-11, err


# --- Structural findings ------------------------------------------------------

def test_storage_is_monotone_in_tolerance(reports):
    bits = [reports[t].bits_after for t in (1e-4, 1e-6, 1e-8, 1e-10)]
    assert bits == sorted(bits), bits


def test_scalars_are_cheaper_than_the_state(reports):
    """alpha and beta only steer; r and z accumulate the answer."""
    for tol in (1e-4, 1e-6, 1e-8):
        r = reports[tol]
        for scalar in ("alpha", "beta"):
            assert rank(fmt(r, scalar)) <= rank(fmt(r, "r")), (tol, scalar, r.mixed)


def test_search_direction_is_cheaper_than_the_residual(reports):
    for tol in (1e-4, 1e-8):
        r = reports[tol]
        assert rank(fmt(r, "p")) <= rank(fmt(r, "r")), (tol, r.mixed)


def test_matrix_storage_drops_below_fp64_at_loose_tolerance(reports):
    """The bandwidth-relevant result: fp32 storage suffices at 1e-6."""
    for tol in (1e-4, 1e-6):
        assert rank(fmt(reports[tol], "A")) <= rank(FP32), reports[tol].mixed


def test_tight_tolerance_forces_full_precision(reports):
    """Matching the fp64 iterate to 1e-10 means reproducing its rounding path."""
    assert all(f == FP64.name for f in reports[1e-10].mixed.values())


# --- Prefilter trade on a coupled kernel --------------------------------------

@pytest.mark.parametrize("tol", [1e-4, 1e-8])
def test_prefilter_is_much_cheaper_for_little_quality(tol, reports):
    with_ = tune(kernel, tol=tol, prefilter=True, cases=CASES)
    without = reports[tol]
    assert with_.total_probes * 4 < without.total_probes
    assert with_.bits_after <= without.bits_after * 1.10
