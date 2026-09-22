"""Uniform-precision bisection: the cheap baseline every tuning run should start with.

Before searching a 2^n configuration space, ask the one-dimensional question:
what is the lowest format that works for *everything*?  It costs O(log |ladder|)
evaluations and it brackets the answer.  If uniform fp32 already passes, a mixed
configuration can only be justified by a performance argument, not an accuracy
one -- and if uniform fp64 barely passes, no amount of search will save you.
"""

from __future__ import annotations

from typing import Callable

from ..formats import Format, rank
from ..types import Config, ConfigSpace, Result, uniform

__all__ = ["UniformBisect"]


class UniformBisect:
    """Binary search for the lowest uniform format meeting the goal.

    Assumes accuracy is monotone in precision along the ladder.  That assumption
    can fail on programs where a coarser format gets lucky; `results` keeps every
    observation so a caller can check.
    """

    name = "uniform"

    def __init__(self, ladder: tuple[Format, ...] | None = None):
        self.ladder = ladder
        self.probes = 0
        self.results: dict[Format, Result] = {}

    def search(
        self,
        space: ConfigSpace,
        evaluate: Callable[[Config], Result],
    ) -> dict[str, Format]:
        ladder = self.ladder or space.ladder

        def passes(i: int) -> bool:
            fmt = ladder[i]
            if fmt not in self.results:
                self.probes += 1
                self.results[fmt] = evaluate(uniform(space, fmt))
            return self.results[fmt].passed

        if not passes(0):
            # Even the top of the ladder misses the goal: the goal is wrong, or
            # the algorithm is unstable at any precision the ladder offers.
            return uniform(space, ladder[0])

        lo, hi = 0, len(ladder) - 1  # lo passes; hi unknown
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if passes(mid):
                lo = mid
            else:
                hi = mid - 1
        return uniform(space, ladder[lo])

    def floor(self) -> Format | None:
        """The lowest format observed to pass, or None if nothing did."""
        winners = [f for f, r in self.results.items() if r.passed]
        return min(winners, key=rank) if winners else None
