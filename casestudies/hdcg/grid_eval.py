#!/usr/bin/env python3
"""Parsimony evaluator for Grid's HDCG preconditioner, and the search that drives it.

One solve is minutes on four A100s, so the design is shaped by evaluation cost
rather than by elegance:

* One binary serves every configuration.  The formats arrive as command-line
  arguments, so no probe costs a rebuild (`Caps(needs_rebuild=False)`).
* Every probe is memoized and appended to a JSONL log.  The prefilter and
  ddmin revisit configurations, an allocation can die mid-search, and neither
  should cost a re-solve.
* The whole search runs inside one allocation as a Python loop calling `srun`,
  so queue time is paid once rather than once per solve.

The observable is outer HDCG iterations at fixed tolerance, judged by the
driver's independent fp64 true residual -- not relative error against an fp64
run.  See casestudies/newton/DESIGN.md section 1 for why error is the wrong
question for an iterative method, and NOTES.md for what this study does and
does not measure (storage, not arithmetic).

Usage, from inside an allocation::

    python3 grid_eval.py --binary $SCRATCH/Grid/tests/debug/Test_hdcg_parsimony \
        --grid 16.16.16.32 --mpi 1.1.1.4 --cases 3 --out results-hdcg.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from parsimony.criteria import Criterion, IterationBudget  # noqa: E402
from parsimony.formats import BF16, FP16, FP32, TF32, Format, virtual  # noqa: E402
from parsimony.searches.ddmin import PrecisionDDMin  # noqa: E402
from parsimony.searches.prefilter import SoloSensitivity  # noqa: E402
from parsimony.types import Caps, Config, ConfigSpace, Result, total_bits, uniform  # noqa: E402

__all__ = ["VARIABLES", "LADDER", "STORAGE_RUNGS", "GridHDCGEvaluator", "RoundingNoOp"]

#: The preconditioner inputs the driver can round.  The fine operator is never
#: rounded, which is what lets the true residual stay an independent fp64 check.
VARIABLES = ("subspace", "stencil", "evec")

#: Virtual widths: float64's exponent range, k significand bits.  The study asks
#: "how many bits", so the search runs on this ladder; the named formats below
#: are evaluated afterwards for the headline, where the answer has to name a
#: storage layout that hardware actually has.
LADDER = tuple(virtual(k) for k in (53, 40, 30, 24, 16, 11, 8))

#: Reported, not searched: real formats at (or near) the knee.
STORAGE_RUNGS = (FP32, TF32, FP16, BF16)


class RoundingNoOp(RuntimeError):
    """A variable was asked to round and the driver reported no change.

    This is the failure that quietly invalidates the whole study -- a hook
    attached to the wrong object, or attached after the value is consumed,
    produces a flat curve indistinguishable from "this input is free".  The
    driver reports the relative L2 change it caused; a zero where a change was
    requested stops the search rather than being reported as a knee.
    """


@dataclass
class Probe:
    """One solve of one case."""

    case: int
    converged: bool
    iterations: int
    true_resid: float
    t_setup: float
    t_solve: float
    delta: dict[str, float] = dc_field(default_factory=dict)
    raw: dict = dc_field(default_factory=dict)


def _spec(fmt: Format) -> str:
    """The format name as the C++ side parses it (round_bits.h)."""
    return fmt.name


def _key(config: Config) -> str:
    return ",".join(f"{v}={_spec(config[v])}" for v in sorted(config))


class GridHDCGEvaluator:
    """Run Grid's HDCG driver under a precision configuration and score it."""

    def __init__(
        self,
        binary: str | Path,
        grid_args: Sequence[str] = (),
        launcher: Sequence[str] = ("srun",),
        cases: Sequence[int] = (0, 1, 2),
        workdir: str | Path = ".",
        criterion: Criterion | None = None,
        ladder: tuple[Format, ...] = LADDER,
        timeout: float = 3600.0,
        log: str | Path | None = "probes.jsonl",
        cache: str | Path | None = "probe-cache.json",
        check_rounding: bool = True,
    ):
        self.binary = str(binary)
        self.grid_args = list(grid_args)
        self.launcher = list(launcher)
        self.cases = list(cases)
        self.workdir = Path(workdir)
        self.ladder = ladder
        self.timeout = timeout
        self.check_rounding = check_rounding
        self.criterion: Criterion = criterion or IterationBudget(slack=0.0)

        self.caps = Caps(formats=ladder, needs_rebuild=False, sees_gpu=True,
                         measures_time=True, slowdown=1.0)
        self.probes = 0
        self.solves = 0  # probes that actually ran the binary
        self.log = self.workdir / log if log else None
        self.cache_path = self.workdir / cache if cache else None
        self._cache: dict[str, dict] = {}
        self._space: ConfigSpace | None = None
        self._reference: dict[str, object] | None = None

        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    # ---- running one configuration -------------------------------------

    def _command(self, config: Config, case: int, json_path: Path) -> list[str]:
        cmd = list(self.launcher) + [self.binary] + list(self.grid_args)
        for v in VARIABLES:
            if v in config:
                cmd += [f"--round-{v}", _spec(config[v])]
        cmd += ["--case", str(case), "--json", str(json_path)]
        return cmd

    def _run_case(self, config: Config, case: int) -> Probe:
        out = self.workdir / f"hdcg-case{case}.json"
        if out.exists():
            out.unlink()  # never read a stale file as this probe's answer
        cmd = self._command(config, case, out)

        started = time.time()
        proc = subprocess.run(cmd, cwd=self.workdir, capture_output=True,
                              text=True, timeout=self.timeout)
        elapsed = time.time() - started

        if self.log:
            with self.log.open("a") as fh:
                fh.write(json.dumps({
                    "config": _key(config), "case": case, "cmd": cmd,
                    "returncode": proc.returncode, "wall": elapsed,
                    "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-2000:],
                }) + "\n")

        # Positive verification: the JSON must exist and say it converged.  A
        # non-zero exit with a written JSON is still a data point (divergence
        # is an outcome); a missing JSON is a broken run.
        if not out.exists():
            raise RuntimeError(
                f"driver wrote no JSON (rc={proc.returncode})\n"
                f"cmd: {' '.join(cmd)}\n{proc.stderr[-2000:]}"
            )
        raw = json.loads(out.read_text())
        return Probe(
            case=case,
            converged=bool(raw.get("converged", False)),
            iterations=int(raw.get("iterations", -1)),
            true_resid=float(raw.get("true_resid", math.inf)),
            t_setup=float(raw.get("t_setup", 0.0)),
            t_solve=float(raw.get("t_solve", 0.0)),
            delta=dict(raw.get("delta", {})),
            raw=raw,
        )

    def _verify_rounding(self, config: Config, probe: Probe) -> None:
        """A variable asked to round must report a non-zero change."""
        for v in VARIABLES:
            fmt = config.get(v)
            if fmt is None or fmt.precision >= 53:
                continue
            if v not in probe.delta:
                raise RoundingNoOp(
                    f"{v} at {fmt.name}: driver reported no delta for it; "
                    "the hook is missing or the JSON contract drifted"
                )
            if probe.delta[v] == 0.0:
                raise RoundingNoOp(
                    f"{v} at {fmt.name}: rounding changed nothing "
                    f"(case {probe.case}).  The hook is not on the value the "
                    "solve consumes -- fix it before trusting any knee."
                )

    def _outputs(self, config: Config) -> dict[str, object]:
        outputs: dict[str, object] = {}
        for i, case in enumerate(self.cases):
            p = self._run_case(config, case)
            if self.check_rounding:
                self._verify_rounding(config, p)
            outputs[f"case{i}.converged"] = p.converged
            outputs[f"case{i}.iterations"] = p.iterations
            outputs[f"case{i}.true_resid"] = p.true_resid
            outputs[f"case{i}.t_setup"] = p.t_setup
            outputs[f"case{i}.t_solve"] = p.t_solve
            for v, d in p.delta.items():
                outputs[f"case{i}.delta_{v}"] = d
        return outputs

    def _cached_outputs(self, config: Config) -> dict[str, object]:
        key = _key(config)
        if key not in self._cache:
            self._cache[key] = self._outputs(config)
            self.solves += 1
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self._cache, indent=1))
        return self._cache[key]

    # ---- the Evaluator protocol ----------------------------------------

    def discover(self) -> ConfigSpace:
        """One reference solve with rounding off, to calibrate the criterion."""
        if self._space is None:
            reference = self._cached_outputs({v: self.ladder[0] for v in VARIABLES})
            self.criterion.calibrate(reference)
            self._reference = reference
            self._space = ConfigSpace(variables=list(VARIABLES), ladder=self.ladder)
        return self._space

    def evaluate(self, config: Config) -> Result:
        if self._space is None:
            self.discover()
        self.probes += 1
        full = {v: self.ladder[0] for v in VARIABLES}
        full.update(config)
        outputs = self._cached_outputs(full)
        passed, score, meta = self.criterion.accept(outputs)
        seconds = sum(float(outputs[f"case{i}.t_solve"]) for i in range(len(self.cases)))
        return Result(dict(full), float(score), bool(passed), seconds=seconds, meta=meta)

    def recalibrate(self, criterion: Criterion) -> None:
        """Swap the criterion (a different slack) without re-solving anything.

        The cache holds outputs, not verdicts, so a slack sweep costs no GPU
        time beyond the first pass.
        """
        assert self._reference is not None, "call discover() first"
        criterion.calibrate(self._reference)
        self.criterion = criterion


# ---- the search ---------------------------------------------------------


def solo_floors(ev: GridHDCGEvaluator, space: ConfigSpace) -> tuple[ConfigSpace, dict]:
    pre = SoloSensitivity(ev.ladder)
    narrowed = pre.analyze(space, ev.evaluate)
    return narrowed, {"floors": pre.report(), "probes": pre.probes}


def run(ev: GridHDCGEvaluator, slacks: Sequence[float]) -> dict:
    """Prefilter, then ddmin, at each slack.  Shares one cache across slacks."""
    space = ev.discover()
    reference = {k: v for k, v in (ev._reference or {}).items()}
    records = []

    for slack in slacks:
        ev.recalibrate(IterationBudget(slack=slack))
        narrowed, pre = solo_floors(ev, space)
        search = PrecisionDDMin(ev.ladder)
        mixed = search.search(narrowed, ev.evaluate)
        final = ev.evaluate(mixed)
        records.append({
            "slack": slack,
            "floors": pre["floors"],
            "prefilter_probes": pre["probes"],
            "search_probes": search.probes,
            "mixed": {v: f.name for v, f in sorted(mixed.items())},
            "bits_before": total_bits(uniform(space, ev.ladder[0])),
            "bits_after": total_bits(mixed),
            "passed": final.passed,
            "worst_iteration_ratio": final.error,
            "iterations": final.meta.get("iterations", {}),
            "trace": search.trace,
        })
        print(f"slack {slack:>5}: "
              + " ".join(f"{v}={records[-1]['mixed'][v]}" for v in VARIABLES)
              + f"  ratio={final.error:.3f} passed={final.passed}")

    return {
        "reference": reference,
        "variables": list(VARIABLES),
        "ladder": [f.name for f in ev.ladder],
        "cases": ev.cases,
        "probes": ev.probes,
        "solves": ev.solves,
        "results": records,
    }


def storage_sweep(ev: GridHDCGEvaluator) -> list[dict]:
    """Each variable alone at each real storage format -- the headline table."""
    rows = []
    for v in VARIABLES:
        for fmt in STORAGE_RUNGS:
            cfg = {u: ev.ladder[0] for u in VARIABLES}
            cfg[v] = fmt
            try:
                r = ev.evaluate(cfg)
            except RoundingNoOp as exc:
                rows.append({"variable": v, "format": fmt.name, "error": str(exc)})
                continue
            rows.append({"variable": v, "format": fmt.name, "passed": r.passed,
                         "worst_iteration_ratio": r.error,
                         "iterations": r.meta.get("iterations", {})})
            print(f"  {v:>9} {fmt.name:>6}: ratio={r.error:.3f} passed={r.passed}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="Test_hdcg_parsimony")
    ap.add_argument("--grid", default="16.16.16.32")
    ap.add_argument("--mpi", default="1.1.1.4")
    ap.add_argument("--launcher", default="srun",
                    help="space-separated launcher, e.g. 'srun -n 4'")
    ap.add_argument("--cases", type=int, default=3, help="number of RNG seeds")
    ap.add_argument("--slacks", default="0,0.1,0.25")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--out", default="results-hdcg.json")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--storage-sweep", action="store_true",
                    help="also run each variable at fp32/tf32/fp16/bf16")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="arguments passed through to the driver")
    args = ap.parse_args(argv)

    grid_args = ["--grid", args.grid, "--mpi", args.mpi] + list(args.extra)
    ev = GridHDCGEvaluator(
        binary=args.binary,
        grid_args=grid_args,
        launcher=args.launcher.split(),
        cases=list(range(args.cases)),
        workdir=args.workdir,
        timeout=args.timeout,
    )

    slacks = [float(s) for s in args.slacks.split(",") if s.strip()]
    report = run(ev, slacks)
    if args.storage_sweep:
        print("storage formats, one variable at a time:")
        report["storage_sweep"] = storage_sweep(ev)

    out = Path(args.workdir) / args.out
    out.write_text(json.dumps(report, indent=1))
    print(f"\n{ev.probes} probes, {ev.solves} solves -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
