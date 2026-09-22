"""parsimony -- find the fewest bits a computation actually needs.

Successor in spirit to Precimonious (Rubio-Gonzalez et al., SC'13) and its
Blame Analysis follow-on (ICSE'16), rebuilt around two pluggable axes --
evaluators and search strategies -- and around the formats that matter now
(bf16, fp8, tf32) rather than the fp32/fp64/fp80 lattice of 2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from .formats import (
    BF16, DEFAULT_LADDER, FP8_E4M3, FP8_E5M2, FP16, FP32, FP64, TF32,
    Format, round_to, virtual,
)
from .evaluators.numpy_eval import NumpyEvaluator, Precision
from .searches.ddmin import PrecisionDDMin, ddmin
from .searches.uniform import UniformBisect
from .types import Caps, Config, ConfigSpace, Goal, Result, summarize, total_bits, uniform

__version__ = "0.1.0"

__all__ = [
    "Format", "FP64", "FP32", "TF32", "BF16", "FP16", "FP8_E5M2", "FP8_E4M3",
    "DEFAULT_LADDER", "virtual", "round_to",
    "Caps", "Goal", "Result", "ConfigSpace", "Config",
    "NumpyEvaluator", "Precision",
    "PrecisionDDMin", "ddmin", "UniformBisect",
    "tune", "TuneReport",
]


@dataclass
class TuneReport:
    """What a tuning run found, in a form that fits in a JSON blob."""

    tolerance: float
    uniform_floor: str | None
    uniform_probes: int
    mixed: dict[str, str]
    mixed_counts: dict[str, int]
    mixed_probes: int
    mixed_error: float
    variables: int
    bits_before: int
    bits_after: int
    trace: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def bit_reduction(self) -> float:
        return 1.0 - self.bits_after / self.bits_before if self.bits_before else 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["bit_reduction"] = round(self.bit_reduction, 4)
        return d


def tune(
    kernel: Callable[[Precision], Mapping[str, object]],
    tol: float = 1e-6,
    ladder: tuple[Format, ...] = DEFAULT_LADDER,
) -> TuneReport:
    """Tune `kernel` to the lowest precision meeting a relative-error tolerance.

    Runs the uniform baseline first, then the ddmin mixed-precision search, and
    reports both -- the gap between them is the only honest measure of whether
    the expensive search bought anything.
    """
    ev = NumpyEvaluator(kernel, Goal(rel_tol=tol), ladder)
    space = ev.discover()

    bisect = UniformBisect(ladder)
    bisect.search(space, ev.evaluate)
    floor = bisect.floor()

    search = PrecisionDDMin(ladder)
    mixed = search.search(space, ev.evaluate)
    final = ev.evaluate(mixed)

    return TuneReport(
        tolerance=tol,
        uniform_floor=floor.name if floor else None,
        uniform_probes=bisect.probes,
        mixed={v: f.name for v, f in sorted(mixed.items())},
        mixed_counts=summarize(mixed),
        mixed_probes=search.probes,
        mixed_error=final.error,
        variables=len(space.variables),
        bits_before=total_bits(uniform(space, ladder[0])),
        bits_after=total_bits(mixed),
        trace=search.trace,
    )
