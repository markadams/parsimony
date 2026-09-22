"""Acceptance criteria: what it means for a configuration to pass.

The CG study showed that the criterion is not a detail.  Relative error against
a full-precision run is right for a function that computes a value and wrong for
an iterative method, where near the solver's own convergence level it measures
bit-reproduction of a rounding trajectory rather than solution quality, and where
for rate-limiting quantities (a Jacobian, a preconditioner) it measures nothing
at all, since they cannot change the fixed point.

So the criterion is pluggable.  Every criterion sees the outputs of the reference
run once, at `calibrate`, and then judges each candidate's outputs with `accept`.
With several cases the output names are prefixed ``case{i}.``; criteria that need
per-case structure use `by_case` to recover it.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Protocol

import numpy as np

__all__ = ["Criterion", "RelativeError", "IterationBudget", "by_case"]

_CASE = re.compile(r"^case(\d+)\.(.+)$")


def by_case(outputs: Mapping[str, object]) -> dict[int, dict[str, object]]:
    """Group ``case{i}.name`` keys by case; un-prefixed keys become case 0."""
    grouped: dict[int, dict[str, object]] = {}
    for key, value in outputs.items():
        m = _CASE.match(key)
        idx, name = (int(m.group(1)), m.group(2)) if m else (0, key)
        grouped.setdefault(idx, {})[name] = value
    return grouped


class Criterion(Protocol):
    def calibrate(self, reference: Mapping[str, object]) -> None: ...

    def accept(self, outputs: Mapping[str, object]) -> tuple[bool, float, dict]:
        """Return (passed, score, meta).  Lower score is better; inf is failure."""
        ...


class RelativeError:
    """Largest relative error over all named outputs, against the reference.

    Falls back to absolute error where the reference is zero, so an output that
    should be exactly zero cannot hide a wrong answer behind a division.
    """

    def __init__(self, tol: float):
        self.tol = tol
        self._reference: dict[str, np.ndarray] = {}

    def calibrate(self, reference: Mapping[str, object]) -> None:
        self._reference = {k: np.asarray(v, dtype=np.float64)
                           for k, v in reference.items()}

    def accept(self, outputs: Mapping[str, object]) -> tuple[bool, float, dict]:
        worst = 0.0
        for name, ref in self._reference.items():
            if name not in outputs:
                return False, math.inf, {"missing": name}
            got = np.asarray(outputs[name], dtype=np.float64)
            if got.shape != ref.shape or not np.all(np.isfinite(got)):
                return False, math.inf, {"invalid": name}
            denom = np.where(ref == 0.0, 1.0, np.abs(ref))
            worst = max(worst, float(np.max(np.abs(got - ref) / denom)))
        return worst <= self.tol, worst, {}


class IterationBudget:
    """Converged, within a multiple of the reference's iteration count.

    The kernel must return ``converged`` (truthy) and ``iterations`` (int) per
    case.  A candidate passes if every case converged and none took more than
    ``ceil((1 + slack) * reference_iterations)``.  The score is the worst ratio
    of candidate to reference iterations, so a passing configuration at slack 0
    scores exactly 1.0.

    This is the criterion for iterative methods: accuracy is fixed by the
    solver's own stopping test, and precision trades against cost.
    """

    def __init__(self, slack: float = 0.0,
                 converged_key: str = "converged", iterations_key: str = "iterations"):
        self.slack = slack
        self.converged_key = converged_key
        self.iterations_key = iterations_key
        self._reference_iterations: dict[int, int] = {}

    def calibrate(self, reference: Mapping[str, object]) -> None:
        self._reference_iterations = {}
        for idx, outs in by_case(reference).items():
            if not outs.get(self.converged_key, False):
                raise ValueError(f"reference run did not converge in case {idx}")
            self._reference_iterations[idx] = int(outs[self.iterations_key])

    def accept(self, outputs: Mapping[str, object]) -> tuple[bool, float, dict]:
        grouped = by_case(outputs)
        worst, iters = 0.0, {}
        for idx, n_ref in self._reference_iterations.items():
            outs = grouped.get(idx, {})
            if not outs.get(self.converged_key, False):
                return False, math.inf, {"diverged_case": idx}
            n = int(outs[self.iterations_key])
            iters[idx] = n
            worst = max(worst, n / n_ref)
        budget = 1.0 + self.slack
        # ceil on the count, not the ratio, so a budget of 0 admits exactly n_ref.
        passed = all(iters[i] <= math.ceil(budget * self._reference_iterations[i])
                     for i in iters)
        return passed, worst, {"iterations": iters}
