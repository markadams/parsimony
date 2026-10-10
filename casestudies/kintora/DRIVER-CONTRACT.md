# KINTORA driver contract for the Parsimony precision study

Written 2026-10-10 (Opus 5.5).  Audience: the KINTORA session that implements Stage A of `PLAN.md`, and the Parsimony
session that writes `kintora_eval.py` against it.  This is the interface between the two codes; neither side changes
it alone.  It is also the first written form of the contract Parsimony's `NativeEvaluator` / `CoprocessEvaluator`
(library items P2, P3) will expect of any native driver.  Schema name: `parsimony-driver/1`.

## Options (ex12, read with the solve gate `-eul_mg_solve`)

| option | meaning |
|---|---|
| `-eul_round_<var> <fmt>[,<fmt>...]` | storage format of variable `<var>`, one entry per matrix-free level, **finest first**; a single entry applies to every level |
| `-eul_json <path>` | write the result document below to `<path>` at the end of the run (one-shot mode) |
| `-eul_coprocess` | co-process mode, below |
| `-eul_mg_cycles <N>` | existing: the stationary V-cycle's contraction factor (the `vcycle_rate` field); the study uses 20 |

`<fmt>`: `v53` (rounding off) and `v1` .. `v52`, `fp64`, `fp32`, `tf32`, `bf16`, `fp16`, `fp8e5m2`, `fp8e4m3`, as
`Parsimony::parse_round_spec` in `round_bits.h` accepts them.  **An unknown format aborts the run** (a typo in a sweep
must not run at fp64).  A list with more entries than matrix-free levels aborts too.

`<var>` and its hook site (`src/kin/eul/kokkos/kin_eulop.kokkos.cxx` unless noted):

| var | rounded where | levels |
|---|---|---|
| `dinv` | the 24 x 24 inverse at its store in the `KinEulOpDevPBSetUp24` kernel (`Od[...] = Dk(idx)`) | all matrix-free levels |
| `coup` | the face couplings at their store in `KinEulOpDevPBSetUp` (`G[...] = t`) | all matrix-free levels |
| `kmf` | the `d_mf` tables after `KinEulOpDevMFSetUp` fills them | **below the finest only**: the finest level's tables are the outer operator's (PLAN.md, "The rule") |
| `prol` | `d_pblk` after `KinEulOpDevProlongSetUp` copies it to the device | every fine level of a transfer |
| `vx` | the smoother's iterate after each Richardson step (`EulMFLinesRichardson`, `kin_eulmg.cxx`) and the prolonged correction | all matrix-free levels |
| `vr` | the residual (`EulMFResidual`, `kin_eulmg.cxx`) and the restricted coarse right-hand side | all matrix-free levels |

Rules for the hooks:

- One call per store site (`Od[i] = EulRound(value, spec)` with `spec` the level's `RoundSpec`), so a rewrite of the
  kernel around it (session 8, item 2) carries the hook along.  `spec.identity()` short-circuits: rounding off
  costs one branch.
- When `dinv` is rounded, store into the double array (`d_sw24`) and require `-eul_mg_sweep_fp32 0`; the hook must see
  the fp64 inverse.  With `-eul_round_dinv fp32` the result must equal `-eul_mg_sweep_fp32 1` bitwise (gate 2).
- Set-up-time tables (`kmf`, `prol`) are rounded from a pristine fp64 copy kept on the device, never re-rounded in
  place: in co-process mode a second configuration must start from the exact values (restore, round, solve).
- Each hook reports `delta[var]` = max over levels and mu rows of |round(Q) - Q|_2 / |Q|_2 for the array it rounded
  (a Kokkos `parallel_reduce` at the store).  0 is reported, not omitted, when nothing changed.

## The result document

One JSON object per configuration.  Field names are the contract; extra fields are allowed and ignored.

```json
{
  "schema": "parsimony-driver/1",
  "code": {"name": "kintora-ex12", "commit": "7f8c9d9", "dirty": false,
           "options": "<the full option string>", "backend": "Cuda", "device": "NVIDIA A100-SXM4-40GB"},
  "config": {"dinv": ["v24", "v24", "v24", "v24"], "coup": ["v53"], "kmf": ["v53"],
             "prol": ["v53"], "vx": ["v53"], "vr": ["v53"]},
  "levels": {"fine": 4, "mf": [4, 3, 2, 1], "unknowns": [1179648, 294912, 73728, 18432]},
  "delta": {"dinv": 2.9e-08, "coup": 0.0, "kmf": 0.0, "prol": 0.0, "vx": 0.0, "vr": 0.0},
  "cases": [
    {"id": "row23.dt0.400", "row": 23, "mu": 0.36, "dt": 0.4, "v_nseg": 12,
     "converged": true, "reason": "CONVERGED_RTOL", "iterations": 3, "rtol": 1e-10,
     "true_residual": 6.1e-11, "vcycle_rate": 0.0123, "lu_error": null,
     "t_setup": 0.05, "t_solve": 0.07}
  ],
  "memory": {"eul_sw24": 226492416, "eul_sweepg": 113246208},
  "blocks": {"dinv_cond": {"row23.dt0.400": {"min": 1.0, "median": 3.2, "max": 41.0}}}
}
```

- `config` echoes the formats as **applied**, expanded to one entry per level.  The evaluator checks it against
  what it asked for; a mismatch is a broken run.
- `true_residual`: |b - J x|_2 / |b|_2 after the solve, J unrounded, computed explicitly (one MatMult), not
  FGMRES's recurrence.
- `vcycle_rate`: the mean contraction factor of the stationary V-cycle (`-eul_mg_cycles`), `null` when not run.
- `lu_error`: |x - x_LU|_inf / |x_LU|_inf where the gate runs its LU reference, else `null`.
- `memory`: bytes per `KinDevArray` by name (`-kin_mem_view`'s numbers), optional.
- `blocks.dinv_cond`: the 2-norm or 1-norm condition number (say which in a `"norm"` key) of the 24 x 24 blocks of
  one level's `dinv`, summarized per case, optional (Stage B).
- A diverged or stalled solve is a result (`converged: false`), not an error.  The document is written in every case
  where the run reached the end; **a missing document is a broken run** and stops the search.

## Co-process mode (`-eul_coprocess`)

Set-up (meshes, geometry, the hierarchy, `d_mf`, the transfers) happens once.  Then ex12 loops:

1. Read one line from stdin: a JSON object `{"config": {<var>: [<fmt>, ...], ...}, "id": <int>}`, or `{"quit": true}`.
2. Restore every set-up-time table from its pristine copy, apply the configuration, invalidate the per-row set-up
   (so `KinEulMGSetUpRow` re-inverts `dinv` and rebuilds `coup` with the new formats).
3. Run every case of the gate (`-eul_mg_rows` x `-eul_mg_dts`).
4. Write the result document on **one line** to stdout, prefixed `PARSIMONY-RESULT ` (the evaluator ignores every
   other line, so PETSc's own output can share the stream), and flush.

The same configuration twice in a row must give bitwise identical documents apart from the times (a restore check
the evaluator runs on its first two probes).

## Gates (Stage A exit)

1. Every variable at `v53`: outer counts and residual histories bitwise equal to the unhooked binary on the 9 base
   cases at L2.
2. `-eul_round_dinv fp32 -eul_mg_sweep_fp32 0` bitwise equal to `-eul_mg_sweep_fp32 1`.
3. Every variable at `v24`: `delta[var] > 0` for each.
4. Co-process: configurations A, B, A in sequence; the two A documents are bitwise equal apart from times.
5. `scripts/check_ex12_eul.sh` green.
