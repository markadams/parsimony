"""Core types and the two plugin protocols.

parsimony separates two concerns that prior tools fused together:

  Evaluator -- given a precision configuration, how accurate (and how fast) is
               the program?  numpy emulation, verificarlo, verrou, a native
               rebuild, ... all answer this differently and at wildly different
               cost.

  Search    -- how do I explore the space of configurations?  ddmin, blame-style
               single-pass attribution, uniform bisection, exhaustive.

Keeping them orthogonal is what lets a cheap evaluator drive an expensive search
during development and an expensive evaluator confirm the answer once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from .formats import DEFAULT_LADDER, Format

__all__ = ["Caps", "Goal", "ConfigSpace", "Result", "Evaluator", "Search", "Config"]

#: A precision assignment: variable name -> format.
Config = Mapping[str, Format]


@dataclass(frozen=True)
class Caps:
    """What an evaluator can and cannot do.

    A search consults these before starting; a request that no installed
    evaluator can serve should fail loudly rather than silently substituting
    something weaker.
    """

    formats: tuple[Format, ...] = DEFAULT_LADDER
    needs_rebuild: bool = False
    sees_gpu: bool = False
    measures_time: bool = False
    slowdown: float = 1.0


@dataclass(frozen=True)
class Goal:
    """Acceptance criterion for a configuration.

    `rel_tol` is the maximum relative error against the full-precision
    reference, matching the error-threshold convention of Precimonious and
    PROMISE so published thresholds (1e-4 ... 1e-10) carry over directly.
    """

    rel_tol: float = 1e-6
    require_speedup: bool = False


@dataclass
class ConfigSpace:
    """The variables available for tuning and the formats each may take."""

    variables: list[str]
    ladder: tuple[Format, ...] = DEFAULT_LADDER
    pinned: dict[str, Format] = field(default_factory=dict)

    def candidates(self) -> list[str]:
        return [v for v in self.variables if v not in self.pinned]


@dataclass
class Result:
    """Outcome of evaluating one configuration."""

    config: dict[str, Format]
    error: float
    passed: bool
    seconds: float | None = None
    meta: dict = field(default_factory=dict)


class Evaluator(Protocol):
    caps: Caps

    def discover(self) -> ConfigSpace:
        """Enumerate tunable variables, typically by one reference run."""
        ...

    def evaluate(self, config: Config) -> Result:
        """Run the program under `config` and score it."""
        ...


class Search(Protocol):
    def search(
        self,
        space: ConfigSpace,
        evaluate: Callable[[Config], Result],
    ) -> dict[str, Format]:
        """Return the lowest-precision configuration the strategy could justify."""
        ...


def uniform(space: ConfigSpace, fmt: Format) -> dict[str, Format]:
    """The configuration that puts every unpinned variable at `fmt`."""
    cfg = {v: fmt for v in space.candidates()}
    cfg.update(space.pinned)
    return cfg


def summarize(config: Config) -> dict[str, int]:
    """Count variables per format, for reporting."""
    counts: dict[str, int] = {}
    for fmt in config.values():
        counts[fmt.name] = counts.get(fmt.name, 0) + 1
    return dict(sorted(counts.items()))


def total_bits(config: Config) -> int:
    """Crude size proxy: summed storage width of the assignment."""
    return sum(1 + f.exp_bits + f.mant_bits for f in config.values())
