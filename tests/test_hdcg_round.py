"""The HDCG rounding kernel must agree with round_to(), not merely resemble it.

Grid applies the configuration in C++; parsimony chooses it in Python.  If the
two rounding rules differ -- by a tie-breaking convention, by where the
subnormal branch starts -- then the bit-knee the study reports is a knee of two
different functions, and nothing in the output would say so.  So the C++ is
compiled here and diffed against round_to() exactly, in hex-float, over
magnitudes spanning the exponent range.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from parsimony.formats import BF16, FP16, FP32, FP8_E4M3, FP8_E5M2, TF32, round_to, virtual

HDCG = Path(__file__).resolve().parents[1] / "casestudies" / "hdcg"
FORMATS = [virtual(k) for k in (53, 40, 30, 24, 16, 11, 8)] + [
    FP32, TF32, BF16, FP16, FP8_E5M2, FP8_E4M3
]


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("hdcg") / "round_bits_probe"
    subprocess.run(
        [cxx, "-O2", "-o", str(exe), str(HDCG / "round_bits_probe.cpp")],
        check=True, capture_output=True,
    )
    return exe


def sample() -> np.ndarray:
    """A few hundred doubles: ordinary magnitudes, then the awkward ones."""
    rng = np.random.default_rng(20260927)
    x = np.concatenate([
        rng.normal(scale=1.0, size=200),
        rng.normal(scale=1e6, size=50),
        rng.normal(scale=1e-6, size=50),
        # Exact ties, where round-half-to-even is the whole question.
        np.ldexp(np.arange(1, 64, dtype=np.float64) * 2 + 1, -4),
        # Exponent extremes: fp16/bf16 overflow and underflow live here.
        np.array([0.0, -0.0, 1e300, -1e300, 1e38, 6.5e4, 1e-40, 5e-8, 2.0 ** -1050]),
    ])
    return x


def run_probe(exe: Path, fmt, values: np.ndarray) -> np.ndarray:
    text = "\n".join(float(v).hex() for v in values) + "\n"
    out = subprocess.run([str(exe), fmt.name], input=text, text=True,
                         capture_output=True, check=True).stdout
    return np.array([float.fromhex(line) for line in out.split()], dtype=np.float64)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_matches_round_to(probe, fmt):
    values = sample()
    got = run_probe(probe, fmt, values)
    want = np.asarray(round_to(values, fmt), dtype=np.float64)

    assert got.shape == want.shape
    # Exact equality, and signed zero must survive: a rounding kernel that
    # flushes -0.0 to +0.0 would pass an allclose check and still be wrong.
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    bad = np.flatnonzero(~same)
    assert bad.size == 0, (
        f"{fmt.name}: {bad.size} mismatches, first at x={values[bad[0]]!r} "
        f"got {got[bad[0]]!r} want {want[bad[0]]!r}"
    )
    assert np.array_equal(np.signbit(got), np.signbit(want))


def test_v53_is_identity(probe):
    values = sample()
    assert np.array_equal(run_probe(probe, virtual(53), values), values)


def test_named_formats_match_hardware(probe):
    """fp32/fp16 have hardware to check against; use it rather than round_to."""
    values = sample()
    assert np.array_equal(run_probe(probe, FP32, values),
                          np.asarray(values, dtype=np.float32))
    with np.errstate(over="ignore"):
        assert np.array_equal(run_probe(probe, FP16, values),
                              np.asarray(values, dtype=np.float16))


def test_bad_format_is_an_error(probe):
    """A typo must fail loudly, not silently run the sweep at fp64."""
    for text in ("v0", "v54", "float32", "", "24x"):
        r = subprocess.run([str(probe), text], input="1.0\n", text=True,
                           capture_output=True)
        assert r.returncode != 0, f"{text!r} was accepted"
