"""Prefilters: cheap analyses that shrink the space before the expensive search.

Blame Analysis (Rubio-Gonzalez et al., ICSE'16) established the pattern --- run a
cheap analysis once, use it to prune the configuration space, then hand the
remainder to delta debugging --- and reported the combination running 9x faster on
average than Precimonious alone, up to 38x.

The prefilter here is *not* Blame Analysis.  Blame performs shadow execution at
the granularity of individual LLVM instructions, executing each one at several
operand precisions and merging the resulting requirements across dynamic
instances.  That costs up to 50x runtime online and exhausted 256 GB of memory in
its offline form on a small benchmark.  `SoloSensitivity` instead works at the
granularity of the variables the search already ranges over, which is coarser,
much cheaper, and sound for the purpose of establishing floors.

Method.  For each variable v, bisect the ladder to find the lowest format at
which lowering *v alone* -- every other variable at full precision -- still meets
the goal.  That becomes v's floor, and variables that cannot be lowered even on
their own are pinned out of the search entirely.  Cost is O(n log L) evaluations
for n variables and a ladder of L rungs, against delta debugging's O(n^2) per
rung.

This is a heuristic, not a bound, and the distinction matters.  The tempting
argument -- if v at format f fails with everything else at full precision, it
cannot pass when other variables are lowered too -- is false.  It assumes
lowering a variable only adds error, but rounding is a projection onto a lattice,
so a coarse format downstream can *absorb* upstream error by snapping a badly
wrong value onto a grid point near the reference.  We observed exactly this: on
the mpow example with a single input, `a` alone at 8 bits fails, yet delta
debugging without the prefilter finds a passing configuration with `a` at 8 bits,
because the output variable quantizes onto 0.5.

Two consequences.  First, a configuration the prefilter excludes may genuinely
have passed, so it can return a wider assignment than an unfiltered search --
measured at 0-16% more bits on mpow, for 9-13x fewer evaluations.  Second, the
configurations it excludes are frequently the ones that pass by accident, so the
restriction tends to improve trustworthiness rather than only cost quality.
Neither effect is guaranteed; report both routes when it matters.
"""

from __future__ import annotations

from typing import Callable

from ..formats import Format, rank
from ..types import Config, ConfigSpace, Result

__all__ = ["SoloSensitivity"]


class SoloSensitivity:
    """Establish a per-variable floor by lowering one variable at a time."""

    name = "solo"

    def __init__(self, ladder: tuple[Format, ...] | None = None):
        self.ladder = ladder
        self.probes = 0
        self.floors: dict[str, Format] = {}

    def analyze(
        self,
        space: ConfigSpace,
        evaluate: Callable[[Config], Result],
    ) -> ConfigSpace:
        """Return a copy of `space` with floors filled in and dead variables pinned."""
        ladder = self.ladder or space.ladder
        top = ladder[0]
        base = {v: top for v in space.variables}
        base.update(space.pinned)

        floors: dict[str, Format] = dict(space.floors)
        pinned: dict[str, Format] = dict(space.pinned)

        for v in space.candidates():
            def passes(i: int, _v=v) -> bool:
                trial = dict(base)
                trial[_v] = ladder[i]
                self.probes += 1
                return evaluate(trial).passed

            # lo always passes (it is the top of the ladder, i.e. the reference).
            lo, hi = 0, len(ladder) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if passes(mid):
                    lo = mid
                else:
                    hi = mid - 1

            floors[v] = ladder[lo]
            if lo == 0:
                # Cannot be lowered even on its own; no combination will help.
                pinned[v] = top

        self.floors = floors
        return ConfigSpace(
            variables=list(space.variables),
            ladder=ladder,
            pinned=pinned,
            floors=floors,
        )

    def report(self) -> dict[str, str]:
        return {v: f.name for v, f in sorted(self.floors.items())}
