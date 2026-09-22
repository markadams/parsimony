"""Delta-debugging search over precision configurations.

Clean-room implementation of two published algorithms:

  * `ddmin` -- Zeller & Hildebrandt's delta-debugging minimization: given a
    predicate that holds for the full set, find a 1-minimal subset for which it
    still holds, in O(n^2) predicate evaluations worst case.

  * `PrecisionDDMin` -- the adaptation introduced by Precimonious (Rubio-Gonzalez
    et al., SC'13) and also used by PROMISE: walk a precision ladder from high to
    low, and at each level use ddmin to find the *minimal* set of variables that
    must stay above that level.  Everything else drops.

The mapping between the two takes a moment to see and is worth stating plainly.
ddmin looks for a minimal subset satisfying a predicate; here the subset is the
set of variables *kept at higher precision* and the predicate is "the accuracy
goal is met".  That predicate is monotone in the right direction -- adding
variables back to high precision never hurts accuracy -- which is exactly the
assumption ddmin needs.  Where monotonicity fails (a lucky cancellation that a
coarser format happens to get right), ddmin still terminates but the result is
only locally minimal, which is true of the original tools as well.

Reference: C. Rubio-Gonzalez et al., "Precimonious: tuning assistant for
floating-point precision", SC'13, doi:10.1145/2503210.2503296.  BSD-3-Clause,
(c) 2013 UC Berkeley.  No code from that project is reproduced here.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from ..formats import Format, rank
from ..types import Config, ConfigSpace, Result

__all__ = ["ddmin", "PrecisionDDMin"]


def _split(items: Sequence[str], n: int) -> list[list[str]]:
    """Partition into `n` near-equal contiguous chunks, dropping empties."""
    size, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        stop = start + size + (1 if i < extra else 0)
        if stop > start:
            chunks.append(list(items[start:stop]))
        start = stop
    return chunks


def ddmin(items: Iterable[str], holds: Callable[[list[str]], bool]) -> list[str]:
    """Return a 1-minimal subset of `items` for which `holds` is still true.

    `holds` must be true for the full set.  The result is 1-minimal: removing
    any single element makes `holds` false.
    """
    c = list(items)
    if not c or holds([]):
        return []

    n = 2
    while len(c) > 1:
        chunks = _split(c, n)

        # Reduce to a subset.
        hit = next((s for s in chunks if holds(s)), None)
        if hit is not None:
            c, n = hit, 2
            continue

        # Reduce to a complement.
        hit = None
        for s in chunks:
            drop = set(s)
            comp = [x for x in c if x not in drop]
            if comp and holds(comp):
                hit = comp
                break
        if hit is not None:
            c, n = hit, max(n - 1, 2)
            continue

        if n >= len(c):
            break
        n = min(2 * n, len(c))

    return c


class PrecisionDDMin:
    """Level-by-level ddmin down a precision ladder.

    At each rung, variables still above that rung are candidates for demotion.
    ddmin finds the minimal set that must stay above; the rest are demoted and
    the search moves to the next rung with the survivors.
    """

    name = "ddmin"

    def __init__(self, ladder: tuple[Format, ...] | None = None):
        self.ladder = ladder
        self.probes = 0
        self.trace: list[tuple[str, int, int]] = []

    def search(
        self,
        space: ConfigSpace,
        evaluate: Callable[[Config], Result],
    ) -> dict[str, Format]:
        ladder = self.ladder or space.ladder
        config: dict[str, Format] = {v: ladder[0] for v in space.candidates()}
        config.update(space.pinned)

        for target in ladder[1:]:
            candidates = [v for v in space.candidates()
                          if rank(config[v]) > rank(target)
                          and space.may_take(v, target)]
            if not candidates:
                continue

            def holds(keep_high: list[str], _t=target, _c=candidates) -> bool:
                keep = set(keep_high)
                trial = dict(config)
                for v in _c:
                    if v not in keep:
                        trial[v] = _t
                self.probes += 1
                return evaluate(trial).passed

            must_stay = set(ddmin(candidates, holds))
            demoted = [v for v in candidates if v not in must_stay]
            for v in demoted:
                config[v] = target
            self.trace.append((target.name, len(demoted), len(candidates)))

        return config
