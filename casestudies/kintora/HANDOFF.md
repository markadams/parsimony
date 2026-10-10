# Handoff: KINTORA case study (workstream F)

Written 2026-10-10 (Opus 5.5), end of the session that started the study.  Read `PLAN.md` next.

## State

- **Parsimony** (`~/Codes/parsimony`, main): this session's work is committed in three commits on top of cc3e76c
  (235a344 the fp8 E4M3 fix, 2608461 the case-study documents and workstream F, d294d1c the paper).  107 tests
  pass.  **Pushed** to origin 2026-10-10 at Mark's choice, together with the 8 Grid/HDCG commits held since
  2026-10-06; origin/main = local main.
- **KINTORA** (`~/Codes/plasma-kinetics`): untouched by this session; HEAD 7f8c9d9 at the start.  Session 8 is active
  there (sweep parallelism, PBSetUp24 rewrite, arc-length rule).
- **Nothing is built** for the study on either side.  No hook, no evaluator, no result.

## What exists

| file | what |
|---|---|
| `PLAN.md` | the plan: stored-data inventory, the rounding rule, observables, cases, stages 0 and A-E, library track P0-P9, paper track, the two-way interaction with KINTORA, decisions of 2026-10-10 |
| `DRIVER-CONTRACT.md` | the interface the KINTORA hooks implement: `-eul_round_<var>`, the result document (`parsimony-driver/1`), the co-process line protocol, five gates |
| `FINDINGS-FOR-KINTORA.md` | K1-K6, sent to KINTORA session 8 on 2026-10-10; outcomes not yet recorded |
| `FABLE-BRIEF.md` | the Fable session: predict each quantity's knee before Stage B measures it |
| `../../paper/kintora-precision.tex` | the paper (16 pp with `\planningtrue`, 13 without); `make` in `paper/` builds both papers |

## Decided this session (Mark)

Title 2X+1V; paper in `paper/`; fp8 E4M3 fixed to OCP E4M3FN (done); `round_bits.h` vendored into KINTORA with its
version line; Stage A by a dedicated KINTORA session after session 8 commits; the analysis goes to Fable, before
Stage B.

## Next, in order

1. K1-K6 were sent to KINTORA session 8 (Mark, 2026-10-10).  When a KINTORA session adopts or rejects one, record
   the ledger entry or commit in `FINDINGS-FOR-KINTORA.md`.
2. **Fable session** (can run now, Mac): `FABLE-BRIEF.md`.
3. **Stage 0** (Opus, Mac, Parsimony only; can run in parallel with 2): P1 (`parsimony.native`, `round_bits.h`
   moved out of `casestudies/hdcg/` with a `get_include()`), P2 (native evaluator base extracted from
   `casestudies/hdcg/grid_eval.py`; its tests unchanged), P4 (`RateBudget`, true-residual guard), P5 (weights on the
   configuration space, a bytes report), and a mock driver speaking `DRIVER-CONTRACT.md`.
4. **Stage A** (KINTORA session, after session 8 commits): start it with "Read
   `~/Codes/parsimony/casestudies/kintora/DRIVER-CONTRACT.md` and implement Stage A of `PLAN.md` there", run from
   `~/Codes/plasma-kinetics` under that repository's rules.
5. Stage B (Mac, L2) once A's gates pass; it tests the Fable predictions.

## Traps

- `rm` is aliased to `rm -i` in this shell: a plain `rm` in a tool call blocks on a prompt.  Use `command rm`.
- The fine multigrid level is the outer operator: never round the finest level's `d_mf` (PLAN.md, "The rule").
- KINTORA's memory report units (GB or GiB) are unchecked; the paper carries a `\todo` for it.
