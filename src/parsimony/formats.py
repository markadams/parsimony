"""Floating-point formats and correctly-rounded emulation on top of float64.

A `Format` is an (exponent bits, significand bits) pair, which covers the IEEE
formats, the vendor formats (bf16, tf32, fp8), and the *virtual* precisions used
by Blame Analysis -- a float64 truncated to k significand bits.

Emulation is exact for any format whose significand and exponent range fit
inside float64, which is every format below.  Formats wider than float64
(fp80, double-double) are out of scope for this evaluator; a native or
verificarlo evaluator handles those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Format",
    "FP64", "FP32", "TF32", "BF16", "FP16", "FP8_E4M3", "FP8_E5M2",
    "DEFAULT_LADDER", "virtual", "round_to", "rank",
]


@dataclass(frozen=True, order=False)
class Format:
    """A binary floating-point format.

    `mant_bits` is the number of *explicit* significand bits, following IEEE
    754 naming: float32 is (8, 23) and carries 24 bits of precision once the
    implicit leading bit is counted.
    """

    name: str
    exp_bits: int
    mant_bits: int

    @property
    def precision(self) -> int:
        """Total significand bits, including the implicit leading bit."""
        return self.mant_bits + 1

    @property
    def emax(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def emin(self) -> int:
        return 1 - self.emax

    @property
    def max_finite(self) -> float:
        return float((2.0 - 2.0 ** -self.mant_bits) * 2.0**self.emax)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


FP64 = Format("fp64", 11, 52)
FP32 = Format("fp32", 8, 23)
TF32 = Format("tf32", 8, 10)
FP16 = Format("fp16", 5, 10)
BF16 = Format("bf16", 8, 7)
FP8_E4M3 = Format("fp8e4m3", 4, 3)
FP8_E5M2 = Format("fp8e5m2", 5, 2)

#: Descending precision.  Search strategies walk this from left to right.
DEFAULT_LADDER = (FP64, FP32, TF32, BF16, FP16, FP8_E5M2, FP8_E4M3)


def virtual(bits: int) -> Format:
    """A float64 truncated to `bits` significand bits, full exponent range.

    This is the precision lattice Blame Analysis used (its ``db_k`` levels were
    decimal digits; bits are the honest unit).  Useful for asking "how many bits
    does this value need?" without committing to a storage format.
    """
    return Format(f"v{bits}", 11, max(bits - 1, 0))


def rank(fmt: Format) -> tuple[int, int]:
    """Sort key: higher is more precise.  Total order over formats."""
    return (fmt.precision, fmt.exp_bits)


def round_to(x, fmt: Format):
    """Round `x` to the nearest value representable in `fmt` (ties to even-ish).

    Handles overflow to infinity and gradual underflow into the subnormal range.
    Accepts scalars or arrays; returns the same shape.
    """
    if fmt.precision >= 53 and fmt.exp_bits >= 11:
        return x

    scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    a = np.asarray(x, dtype=np.float64)

    m, e = np.frexp(a)  # a == m * 2**e with 0.5 <= |m| < 1
    unbiased = e - 1  # IEEE exponent E in a == +-1.f * 2**E

    # Normal path: keep `precision` bits after the binary point of m.
    scale = np.exp2(float(fmt.precision))
    y = np.ldexp(np.round(m * scale) / scale, e)

    # Subnormal path: quantize to the smallest representable step.  Skipped
    # unless something is actually subnormal -- for a full-exponent virtual
    # format the quantum underflows float64 and the division would overflow.
    below = unbiased < fmt.emin
    if np.any(below):
        quantum = np.exp2(float(fmt.emin - fmt.mant_bits))
        y = np.where(below, np.round(a / quantum) * quantum, y)

    y = np.where(np.abs(y) > fmt.max_finite, np.copysign(np.inf, a), y)
    y = np.where(a == 0.0, a, y)
    y = np.where(np.isfinite(a), y, a)  # preserve incoming inf/nan

    return float(y) if scalar else y
