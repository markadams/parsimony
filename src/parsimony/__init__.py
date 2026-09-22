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
from .criteria import Criterion, IterationBudget, RelativeError
from .evaluators.numpy_eval import NumpyEvaluator, Precision
from .searches.ddmin import PrecisionDDMin, ddmin
from .searches.prefilter import SoloSensitivity
from .searches.uniform import UniformBisect
from .types import Caps, Config, ConfigSpace, Goal, Result, summarize, total_bits, uniform

__version__ = "0.1.0"

__all__ = [
    "Format", "FP64", "FP32", "TF32", "BF16", "FP16", "FP8_E5M2", "FP8_E4M3",
    "DEFAULT_LADDER", "virtual", "round_to",
    "Caps", "Goal", "Result", "ConfigSpace", "Config",
    "NumpyEvaluator", "Precision",
    "Criterion", "RelativeError", "IterationBudget",
    "PrecisionDDMin", "ddmin", "UniformBisect", "SoloSensitivity",
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
    search_probes: int
    prefilter_probes: int
    mixed_error: float
    variables: int
    bits_before: int
    bits_after: int
    space_before: int = 0
    space_after: int = 0
    trace: list[tuple[str, int, int]] = field(default_factory=list)
    floors: dict[str, str] = field(default_factory=dict)

    @property
    def total_probes(self) -> int:
        """Every evaluation the tuning run cost, prefilter included."""
        return self.prefilter_probes + self.search_probes

    @property
    def bit_reduction(self) -> float:
        return 1.0 - self.bits_after / self.bits_before if self.bits_before else 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["bit_reduction"] = round(self.bit_reduction, 4)
        d["total_probes"] = self.total_probes
        return d


def tune(
    kernel: Callable[[Precision], Mapping[str, object]],
    tol: float = 1e-6,
    ladder: tuple[Format, ...] = DEFAULT_LADDER,
    prefilter: bool = True,
    cases: object | None = None,
    criterion: Criterion | None = None,
) -> TuneReport:
    """Tune `kernel` to the lowest precision meeting a relative-error tolerance.

    Runs the uniform baseline, optionally the solo-sensitivity prefilter, then
    the ddmin mixed-precision search.  Both comparisons are reported because both
    matter: the gap between the uniform floor and the mixed configuration says
    whether the expensive search bought anything, and `prefilter=False` says what
    the prefilter saved.
    """
    ev = NumpyEvaluator(kernel, Goal(rel_tol=tol), ladder, cases=cases,
                        criterion=criterion)
    space = ev.discover()
    space_before = space.size()

    bisect = UniformBisect(ladder)
    bisect.search(space, ev.evaluate)
    floor = bisect.floor()

    pre = SoloSensitivity(ladder) if prefilter else None
    if pre is not None:
        space = pre.analyze(space, ev.evaluate)

    search = PrecisionDDMin(ladder)
    mixed = search.search(space, ev.evaluate)
    final = ev.evaluate(mixed)

    return TuneReport(
        tolerance=tol,
        uniform_floor=floor.name if floor else None,
        uniform_probes=bisect.probes,
        mixed={v: f.name for v, f in sorted(mixed.items())},
        mixed_counts=summarize(mixed),
        search_probes=search.probes,
        prefilter_probes=pre.probes if pre else 0,
        mixed_error=final.error,
        variables=len(space.variables),
        bits_before=total_bits(uniform(space, ladder[0])),
        bits_after=total_bits(mixed),
        space_before=space_before,
        space_after=space.size(),
        trace=search.trace,
        floors=pre.report() if pre else {},
    )
