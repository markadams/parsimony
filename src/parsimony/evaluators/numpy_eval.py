"""Zero-dependency evaluator: emulate reduced precision in numpy.

The kernel under study is an ordinary Python function that takes a `Precision`
recorder and returns a dict of named outputs::

    def kernel(P):
        a   = P("a", 1.84089642)
        acc = P("acc", a * a)
        return {"res": acc}

Every value passed through `P` becomes a tunable variable.  This is deliberately
the same shape as verificarlo's ``vfc_probe`` so a kernel prototyped here can be
ported to an instrumented C build without rethinking what the variables are.

This evaluator is for algorithms you are designing, or small extracted kernels.
It does not measure time and knows nothing about the compiled program, so it
answers "how many bits does this arithmetic need?" and not "will this be
faster?".  The latter needs an evaluator with `caps.measures_time`.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Sequence

import numpy as np

from ..formats import DEFAULT_LADDER, FP64, Format, round_to
from ..types import Caps, Config, ConfigSpace, Goal, Result

__all__ = ["Precision", "NumpyEvaluator"]


class Precision:
    """Rounds values to the format assigned to each name, recording the names."""

    def __init__(self, config: Config | None = None, default: Format = FP64):
        self._config = dict(config or {})
        self._default = default
        self.seen: list[str] = []
        self._seen: set[str] = set()

    def __call__(self, name: str, value):
        if name not in self._seen:
            self._seen.add(name)
            self.seen.append(name)
        return round_to(value, self._config.get(name, self._default))

    #: Explicit alias, for kernels that read better spelled out.
    round = __call__


class NumpyEvaluator:
    """Score a configuration by relative error against an all-fp64 reference."""

    def __init__(
        self,
        kernel: Callable[..., Mapping[str, object]],
        goal: Goal | None = None,
        ladder: tuple[Format, ...] = DEFAULT_LADDER,
        cases: Sequence[object] | None = None,
    ):
        """`cases`, when given, are passed to the kernel one at a time as a second
        argument, and the tolerance must hold for *every* case.

        Use more than one.  With a single input, the search can satisfy a
        tolerance by coarsening an output until it quantizes onto a lattice point
        that happens to sit near the reference -- a downstream format absorbing
        upstream error rather than the computation being accurate.  Several
        inputs remove that accident, since no one grid point is lucky for all of
        them.  Precimonious notes the need for representative inputs and treats
        generating them as orthogonal; it is not optional.
        """
        self.kernel = kernel
        self.goal = goal or Goal()
        self.ladder = ladder
        self.cases = None if cases is None else list(cases)
        self.caps = Caps(formats=ladder, needs_rebuild=False, sees_gpu=False,
                         measures_time=False, slowdown=1.0)
        self.probes = 0
        self._reference: dict[str, np.ndarray] | None = None
        self._space: ConfigSpace | None = None

    def _run(self, recorder: Precision) -> dict[str, object]:
        if self.cases is None:
            return dict(self.kernel(recorder))
        outputs: dict[str, object] = {}
        for i, case in enumerate(self.cases):
            for name, value in self.kernel(recorder, case).items():
                outputs[f"case{i}.{name}"] = value
        return outputs

    def discover(self) -> ConfigSpace:
        """One reference run at fp64, recording the variables the kernel touches."""
        if self._space is None:
            rec = Precision({}, default=FP64)
            outputs = self._run(rec)
            self._reference = {k: np.asarray(v, dtype=np.float64)
                               for k, v in outputs.items()}
            self._space = ConfigSpace(variables=list(rec.seen), ladder=self.ladder)
        return self._space

    def evaluate(self, config: Config) -> Result:
        if self._reference is None:
            self.discover()
        assert self._reference is not None

        self.probes += 1
        start = time.perf_counter()
        try:
            outputs = self._run(Precision(config, default=FP64))
        except (FloatingPointError, ZeroDivisionError, OverflowError) as exc:
            return Result(dict(config), float("inf"), False,
                          meta={"exception": type(exc).__name__})
        seconds = time.perf_counter() - start

        error = _max_relative_error(self._reference, outputs)
        passed = bool(np.isfinite(error)) and error <= self.goal.rel_tol
        return Result(dict(config), float(error), passed, seconds=seconds)


def _max_relative_error(reference: Mapping[str, np.ndarray],
                        candidate: Mapping[str, object]) -> float:
    """Largest relative error over all named outputs.

    Falls back to absolute error where the reference is zero, so an output that
    should be exactly zero cannot mask a wrong answer behind a division.
    """
    worst = 0.0
    for name, ref in reference.items():
        if name not in candidate:
            return float("inf")
        got = np.asarray(candidate[name], dtype=np.float64)
        if got.shape != ref.shape:
            return float("inf")
        if not np.all(np.isfinite(got)):
            return float("inf")
        denom = np.where(ref == 0.0, 1.0, np.abs(ref))
        worst = max(worst, float(np.max(np.abs(got - ref) / denom)))
    return worst
