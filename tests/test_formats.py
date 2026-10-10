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


# ---- fp8: the OCP 8-bit formats as hardware implements them ---------------


def _ocp_e4m3fn_values() -> np.ndarray:
    """Every finite E4M3FN value, decoded byte by byte from the OCP spec.

    Bias 7; exponent field 0 is subnormal (m/8 * 2**-6); every other exponent
    field is normal, including 1111, whose all-ones significand alone is NaN.
    Independent of `Format`, so it checks the definition, not the code.
    """
    return np.array(sorted(_ocp_e4m3fn_codes()))


def _ocp_e4m3fn_codes() -> dict[float, int]:
    """Finite E4M3FN value -> its significand field (the tie-break reads its last bit)."""
    codes: dict[float, int] = {}
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        e, m = (byte >> 3) & 0xF, byte & 0x7
        if e == 0xF and m == 0x7:
            continue  # NaN
        mag = (m / 8.0) * 2.0**-6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        codes.setdefault(sign * mag, m)  # +0 and -0 compare equal; both have m = 0
    return codes


def test_e4m3_is_ocp_e4m3fn():
    vals = _ocp_e4m3fn_values()
    assert FP8_E4M3.max_finite == 448.0 == vals.max()
    assert FP8_E4M3.emin == -6 and FP8_E4M3.emax == 8
    # Every representable value survives rounding unchanged.
    assert np.array_equal(round_to(vals, FP8_E4M3), vals)


def test_e4m3_rounds_to_nearest_even_like_a_brute_force_search():
    vals, codes = _ocp_e4m3fn_values(), _ocp_e4m3fn_codes()
    rng = np.random.default_rng(448)
    x = np.concatenate([rng.uniform(-460, 460, 3000),
                        rng.normal(scale=2.0**-7, size=1000),
                        (vals[1:] + vals[:-1]) / 2])  # every tie
    got = round_to(x, FP8_E4M3)
    for xi, gi in zip(x, got):
        d = np.abs(vals - xi)
        near = vals[d == d.min()]
        if near.size == 2:  # a tie: the even significand (last mantissa bit 0) wins
            even = [v for v in near if codes[float(v)] % 2 == 0]
            assert len(even) == 1 and gi == even[0], (xi, gi, near)
        else:
            assert gi == near[0], (xi, gi, near)


def test_e4m3_overflow_is_nan_and_has_no_infinity():
    assert round_to(460.0, FP8_E4M3) == 448.0   # below the 464 midpoint
    assert round_to(464.0, FP8_E4M3) == 448.0   # the tie goes to the even 448
    assert np.isnan(round_to(470.0, FP8_E4M3))  # would round to 480: no such value
    assert np.isnan(round_to(-1e6, FP8_E4M3))
    assert np.isnan(round_to(np.inf, FP8_E4M3))
    assert round_to(2.0**-9, FP8_E4M3) == 2.0**-9  # smallest subnormal
