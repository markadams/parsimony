# `Test_hdcg_parsimony.cc`: what to change and where

The driver is a copy of Grid's `tests/debug/Test_general_coarse_hdcg.cc` with
rounding hooks and a machine-readable result. It is not written yet — Grid's
source is not on this machine (see NOTES.md, "Blocked"). This file is the
specification, and it is not aspirational: the command line and the JSON below
are already pinned by `tests/test_hdcg_eval.py`, so a driver that satisfies
them drops straight into the search with nothing else to reconcile.

Copy to `casestudies/hdcg/Test_hdcg_parsimony.cc`, symlink or copy it into
Grid's `tests/debug/` (which `scripts/filelist` scans), and build only that
target: `make -C tests/debug Test_hdcg_parsimony`.

## Command line

Grid consumes its own arguments (`--grid`, `--mpi`, `--accelerator-threads`)
through `Grid_init`; these are additions, parsed with
`GridCmdOptionExists` / `GridCmdOptionPayload`.

| flag | default | meaning |
|---|---|---|
| `--round-subspace <fmt>` | `v53` | format for `Aggregates.subspace[b]` |
| `--round-stencil <fmt>` | `v53` | format for the coarse operator's matrices |
| `--round-evec <fmt>` | `v53` | format for the deflation eigenvectors |
| `--case <n>` | `0` | selects the source RNG seed; `n` is an index, not the seed |
| `--json <path>` | none | write the result record and exit 0 |

`<fmt>` is parsed by `Parsimony::parse_round_spec` (round_bits.h): `v53`…`v1`
for virtual widths, or `fp64`/`fp32`/`tf32`/`bf16`/`fp16`/`fp8e4m3`/`fp8e5m2`.
`v53` is rounding off. **An unrecognized format must abort**, not fall back —
`parse_round_spec` returns false precisely so a typo in a sweep script cannot
quietly produce an fp64 data point labelled `bf16`.

`--case n` must change only the RHS RNG seed. The gauge field, the subspace RNG
and every solver parameter stay fixed, or the cases stop being repeats of one
measurement and the iteration ratios across them mean nothing.

## JSON record

Written once per invocation, to `--json`:

```json
{"converged": true, "iterations": 24, "true_resid": 8.1e-09,
 "t_setup": 112.4, "t_solve": 38.2,
 "delta": {"subspace": 3.2e-13, "stencil": 0.0, "evec": 0.0},
 "case": 0}
```

* `converged` / `iterations` — the outer HDCG count, what `IterationBudget`
  judges. Write the record even when the solve fails: non-convergence is a
  data point, a missing file is a broken run, and the evaluator distinguishes
  them (`test_missing_json_is_an_error_not_a_divergence`).
* `true_resid` — `‖b − A x‖ / ‖b‖` recomputed in fp64 with the *fine*
  operator after the solve. Since only preconditioner inputs are rounded the
  fine operator is untouched, so this needs no special build; it is what stops
  a rounded solver from reporting its own stopping test as success. This is the
  rule the Newton study had to be taught by failing (DESIGN.md §1, last bullet).
* `t_setup` / `t_solve` — seconds. Setup is subspace + coarsen + Lanczos; it is
  repeated per evaluation, so its size decides whether the subspace is worth
  caching (NOTES.md, open question 1).
* `delta` — the relative L2 change each rounding actually caused, as returned
  by `Parsimony::RoundField`. Emit a key for every variable that was rounded.
  The evaluator raises `RoundingNoOp` on a zero, because a hook attached to the
  wrong object, or attached after the value has been consumed, otherwise
  reports as "this input needs no bits" — the one failure mode of this study
  that produces a plausible publishable curve.

## The three hooks

```cpp
#include "round_field.h"   // after <Grid/Grid.h>

Parsimony::RoundSpec spec_subspace = Parsimony::ROUND_OFF, spec_stencil = ..., spec_evec = ...;
std::map<std::string,double> delta;
```

**1. `subspace` — after `Aggregates.Orthogonalise()`, before `CoarsenOperator`.**

```cpp
Aggregates.Orthogonalise();
delta["subspace"] = Parsimony::RoundFields(Aggregates.subspace, spec_subspace);
// CoarsenOperator(...) follows, and sees the rounded vectors.
```

Order matters and is the point: rounding *before* the coarsen keeps the
two-level method Galerkin-consistent, because the coarse operator is built from
the same rounded vectors the projector uses. Rounding after would break
consistency and measure a different thing. Note that orthogonality is lost to
O(2⁻ᵏ) here; that is part of what is being measured, not a bug to repair with a
second Orthogonalise().

**2. `stencil` — after the multi-RHS coarse operator is built.**

```cpp
for (int p = 0; p < geom.npoint; p++) {
  auto A = mrhs_op.GetMatrix(p);                 // BLAS_A[p], public
  delta["stencil"] = std::max(delta["stencil"],
      Parsimony::RoundComplexBuffer(A.data(), A.size(), spec_stencil));
  mrhs_op.SetMatrix(p, A);
}
```

This one *does* break consistency between the coarse operator and the
projector, which is why the plan expects it to knee sharply while `subspace`
degrades gently. Check the accessor signatures against the installed header
before trusting the sketch above — `GetMatrix`/`SetMatrix` were read from the
class declaration, not from a compile.

**3. `evec` — after `IRL.calc`, before `ImportEigenBasis`.**

```cpp
IRL.calc(eval, evec, src, Nconv, false);
delta["evec"] = Parsimony::RoundFields(evec, spec_evec);
Solver.ImportEigenBasis(evec, eval);
```

Only the deflation guess depends on these, so this is the tolerance control:
if `evec` does not go lower than the other two, suspect the harness before the
physics.

`coarse_src` from the plan's table is dropped unless a hook turns out to be a
one-liner. It is the only one of the four whose rounding sits inside the solve
rather than in setup, so it is also the only one that would need the hook to be
cheap.

## Sanity gate before any search

`v53` everywhere must reproduce the unmodified driver's iteration count
*exactly*, and every `delta` must be absent or zero. If it does not, the hooks
are perturbing the run by existing, and nothing downstream is interpretable.
