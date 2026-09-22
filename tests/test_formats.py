"""Emulated rounding must agree with the hardware where hardware exists."""

from __future__ import annotations

import numpy as np
import pytest

from parsimony import BF16, FP8_E4M3, FP16, FP32, FP64, TF32, round_to, virtual


def test_fp64_is_identity():
    for x in [0.1, -3.25, 1e-300, 1e300, 0.0]:
        assert round_to(x, FP64) == x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fp32_matches_hardware(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=1e3, size=4096)
    assert np.array_equal(round_to(x, FP32), np.asarray(x, dtype=np.float32))


def test_fp16_matches_hardware():
    rng = np.random.default_rng(7)
    x = rng.normal(scale=10.0, size=4096)
    assert np.array_equal(round_to(x, FP16), np.asarray(x, dtype=np.float16))


def test_fp16_subnormals_match_hardware():
    x = np.ldexp(np.arange(1, 512, dtype=np.float64), -24)  # below fp16 emin
    assert np.array_equal(round_to(x, FP16), np.asarray(x, dtype=np.float16))


def test_bf16_keeps_fp32_exponent_range():
    # bf16 has fp32's exponent, so a value fp16 overflows on survives.
    big = 1e38
    assert np.isfinite(round_to(big, BF16))
    assert np.isinf(round_to(big, FP16))


def test_overflow_goes_to_infinity():
    assert np.isinf(round_to(1e40, FP16))
    assert np.isinf(round_to(-1e40, FP16))
    assert round_to(-1e40, FP16) < 0


def test_exact_values_survive_every_format():
    for fmt in [FP32, TF32, BF16, FP16, FP8_E4M3]:
        for x in [0.0, 1.0, -2.0, 0.5, 4.0]:
            assert round_to(x, fmt) == x


def test_precision_decreases_down_the_ladder():
    x = np.pi
    errors = [abs(round_to(x, f) - x) for f in (FP32, TF32, BF16, FP8_E4M3)]
    assert errors == sorted(errors), errors


def test_virtual_matches_fp32_significand():
    rng = np.random.default_rng(3)
    x = rng.normal(size=1024)
    assert np.array_equal(round_to(x, virtual(24)), round_to(x, FP32))


def test_nan_and_inf_pass_through():
    assert np.isnan(round_to(float("nan"), FP32))
    assert np.isinf(round_to(float("inf"), FP32))


def test_scalars_stay_scalars():
    assert isinstance(round_to(0.1, FP32), float)
    assert isinstance(round_to(np.arange(3.0), FP32), np.ndarray)
