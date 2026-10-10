# Case study: KINTORA's Eulerian matrix-free multigrid on the A100 -- plan

Written 2026-10-09 (Opus 5.5, from KINTORA Eulerian session 7); revised 2026-10-10 (Opus 5.5) against KINTORA
7f8c9d9 and the session-8 brief (`~/Codes/plasma-kinetics/plans/prompt-eulerian-session8.md`).  Audience: Mark and
the Parsimony and KINTORA sessions that carry it out.  Status: planned; nothing is built on either side.

Three products, one schedule:

1. **The study**: how few bits each stored quantity of KINTORA's V-cycle needs, and what those bits buy.
2. **The paper**: `paper/kintora-precision.tex` (beside `paper/parsimony.tex`, sharing `paper/refs.bib`), "Precimonious and Blame Analysis methods applied to a
   minimal-precision matrix-free multigrid solve ..." -- a case study of the methods on a production GPU solver.
3. **The library**: the pieces of Parsimony that the study needs and the HDCG study improvised (track P below).
   Grid was the first native consumer; KINTORA is the second, and the second consumer is when to extract the
   common base.

## Why this code

KINTORA's Eulerian drift-kinetic solver (`~/Codes/plasma-kinetics`, `src/kin/eul/`; the model is 3X+2V, each stage
solve is 2X+1V: one poloidal plane x v_par at fixed mu) solves each implicit stage with FGMRES (fp64, rtol 1e-10, restart 30, at most 200 its) preconditioned by one V-cycle of a matrix-free h-multigrid
on the GPU (Kokkos, A100).  The defaults (`-eul_mg_device -eul_mg_mf`, KINTORA 7f8c9d9, ledger D230(c)): line kernel
3 (a forward and a backward sweep along each ring arc), (cell, segment) 24 x 24 diagonal blocks, inverses stored fp32,
Richardson V(3,3), arcs of 64 cells on the finest level halved per coarser level, L0 assembled and solved by cuDSS.
At 12 v segments a mu row has 288 unknowns per cell: 1.18M at L4, 4.72M at L5, 18.9M at L6.  Measured:

- L4, 9 cases (mu rows 0/12/23 x dt .05/.2/.4): set-up + solve 0.65 s summed, outer its 2,2,2 / 2,2,3 / 2,2,3.
  L5 ~2.0 s, its 2,2,3 / 2,3,3 / 2,3,4.  L6 on one A100: 3-6 its, 0.55-0.87 s per stage solve, 12.8 GB (D230(c)).
- fp32 inverses: outer counts identical at L4, L5 and L4 v48; L5 device arrays 2.49 -> 1.91 GB; the L4 sweep
  600 -> 490 us.  The sweep is latency-bound (ncu: 780 lines x 24 lanes = 195 teams, 8% occupancy).

So the question is no longer "is fp32 enough" but **how few bits each quantity needs**, and the answer pays in two
currencies that the study must keep apart:

- **Memory**, now: the bytes stored decide the largest mesh one GPU holds (L6 is 10.5-12.8 GB; L7 would be 4x).
- **Time**, later: a latency-bound sweep gains little from fewer bytes (fp32 gave 1.22x on the sweep, for half its
  D^-1 bytes).  Session 8's work on the sweep's parallelism (more lanes per line) is what could make it
  bandwidth-bound, and only then does precision buy time in proportion.  The bit answers do not depend on that
  work; the time answers do, so they are measured after it.

Why it is a good second native case study after HDCG: a preconditioner (so `IterationBudget` is the right criterion
and error against fp64 measures nothing), GPU-resident, blocked data where narrowing values cuts bytes, and a physics
stiffness knob (v resolution, mu) that changes what the smoother must resolve.

## The stored data, ranked by bytes

Values per unknown of one mu row's solve, read from the allocation sizes in `kin_eulop.kokkos.cxx`.  "Levels" are
the matrix-free levels L1..Lfine; their unknowns sum to 1.33x the finest level's.

| name | array | values per unknown | written in | read by | role |
|---|---|---|---|---|---|
| `dinv` | `d_sw24` / `d_sw24f` (24 x 24 per (cell, segment)) | 24 | `KinEulOpDevPBSetUp` (team inversion, the `KinEulOpDevPBSetUp24` kernel) | the sweep | smoother |
| `coup` | `d_sweepg` (two 6 x 6 per node block) | 12 | `KinEulOpDevPBSetUp` | the sweep | smoother |
| `vx`, `vr` | PETSc VECKOKKOS per level (iterate, correction / residual, coarse right-hand side) | ~6 total | PCMG, `EulMFLinesRichardson`, `EulMFResidual` | everything | V-cycle state |
| `kmf` | `d_mf` (`MQ_*`, `MF_*`, `MV_*` tables) | ~2.4 (~690 per cell, shared by all mu rows) | `KinEulOpDevMFSetUp` | `EulDevApplyK` | level operator K x |
| `prol` | `d_pblk` (6 x 6 per fine cell) | 0.125 | `KinEulOpDevProlongSetUp` | prolong / restrict | transfer |
| -- | `d_pbinv` (6 x 6 node blocks) | 6 | `KinEulOpDevPBSetUp` | **set-up only** on the 24 x 24 path (it holds the uninverted node blocks the team kernel reads) | -- |
| -- | `d_Minv` (6 x 6 blocks of M_J^-1) | 6 | `KinEulOpDevSetUp` (every level) | `EulMGJMult` on the finest level | **outer operator: never rounded**; the coarse levels' copies look unused |

Derived, to be confirmed by `-kin_mem_view` in Stage C: at L5, `dinv` over L1..L5 is 4.72M x 1.332 x 24 x 8 B = 1.21 GB
at fp64, so fp32 should save 0.60 GB; D230(c) measured 0.58 GB (2.49 -> 1.91; the report's GB vs GiB is unchecked).
After fp32, `coup` (0.60 GB at fp64) is as large as `dinv`.  **Consequences:**

- Only `dinv`, `coup` and the vectors can move the memory or bandwidth totals.  `kmf` and `prol` are scientifically
  interesting variables (how sensitive is the hierarchy to its operators?) with no payoff, so the search objective
  must weight variables by their bytes (library item P5).  Unweighted, ddmin reports "`prol` at fp8" as a win.
- `d_pbinv` looks like resident set-up scratch on the 24 x 24 path (6 values per unknown, 0.30 GB at L5).  A memory
  item for the KINTORA side, not a precision one -- check that nothing in the solve reads it before freeing it.

## The rule that decides what may be rounded

**Only preconditioner input is rounded.**  The outer operator J = M_J^-1 K and FGMRES stay fp64, so the converged
answer cannot move and pass/fail is by iterations.  This rule has a consequence the first version of this plan
missed: **the finest level of the V-cycle is the outer operator.**  `KinEulMGCreate` sets `mg->op[nlev-1] = fine`,
and `EulMGJMult` applies `mg->K[nlev-1]`, the same MATSHELL that smooths the finest level, reading the same `d_mf`
tables.  Rounding the finest level's `kmf` would change the matrix FGMRES converges on, and it would break the
conservation ledgers, which hold to 1e-16 per step because `F_h . grad H_h` vanishes at every quadrature point.  So:

- `kmf` is rounded on levels below the finest only.
- The finest level's operator inside the V-cycle becomes a separate variable `kmf_fine` only if the coarse-level
  `kmf` turns out insensitive enough to make it worth a second, rounded copy of the tables (~2.4 values per unknown,
  a 6% memory cost against `dinv` + `coup`).  Production would have to store that copy too.
- `dinv`, `coup`, `prol`, `vx` and `vr` are preconditioner-only on every level, the finest included.

## Observables and criteria

- **Outer FGMRES iterations at rtol 1e-10** (`IterationBudget`): what production pays.  Quantized: at 2-4 its a
  budget can only resolve changes of 25-50%.
- **The stationary V-cycle's contraction factor** (`-eul_mg_cycles N`: x += MG(M_J (b - J x)), the mean factor down
  to 1e-12, already printed by the solve gate): continuous, so it resolves degradation that does not yet cost an
  outer iteration.  This is the observable the blame-style analysis needs (P4, P7).
- **An independent fp64 true residual** |b - J x| / |b| after each solve, with J unrounded: the guard against a
  solver stopping on a residual it computed wrongly (the Bratu false-stop, Parsimony paper).
- **|x - x_LU| / |x_LU|** where the gate's LU reference exists (L2, <= 80k unknowns): a control, not a criterion.

## Cases

- Base set: mu rows 0/12/23 x dt .05/.2/.4, 12 v segments (9 cases).  At least 3 rows and 3 dt, because the
  iteration count is quantized and one case can be lucky (Parsimony finding 2).
- **Stiff set: `-v_nseg 48`, mu rows 18/23, dt .2/.4** (4 cases).  At 12 v segments every smoother looks fine; at 48
  and large mu the 6 x 6 node-block sweep stalled (FGMRES flat at 3e-5, ledger D230(b)) while the 24 x 24 sweep
  converges in 2-4.  The v coupling *between* segments is still left to the Richardson iteration (session-8 brief,
  item 7), so the stiff set is where a rounded `dinv` is most likely to show.
- v12 and v48 are different grids, so different driver processes.

## Stages

Each stage ends at a checkpoint: tests green on the Parsimony side, `check_ex12_eul.sh` green on the KINTORA side,
`casestudies/kintora/HANDOFF.md` updated with the tree state of both repositories and the exact next step.

### Stage 0 -- library groundwork (Parsimony, Mac, 1 session, Opus)

P0, P1, P2, P4, P5 of the library track, with a mock driver that speaks the contract
(`casestudies/kintora/DRIVER-CONTRACT.md`) so the evaluator is tested before KINTORA has a hook.  Exit: `pytest`
green, `grid_eval.py` running on the extracted base with its tests unchanged.

### Stage A -- the hooks in KINTORA (a dedicated KINTORA session, Mac, 1 session, Opus)

Run from `~/Codes/plasma-kinetics` under that repository's rules (AGENTS.md, ask before committing), **after session 8
commits**: session 8's item 2 rewrites `KinEulOpDevPBSetUp24`, which is the `dinv` hook site.  The hooks go at the
store sites as one call each (`Od[i] = EulRound(dk, spec)`), so they survive a kernel rewrite.  Spec:
`DRIVER-CONTRACT.md`.  Gates:

1. All formats `v53`: outer counts and residual histories bitwise equal to the unhooked binary.
2. `-eul_round_dinv fp32 -eul_mg_sweep_fp32 0` bitwise equal to `-eul_mg_sweep_fp32 1`.  The production fp32 path
   is a cast to float, read back to double in the sweep, which is exactly storage rounding with fp64 arithmetic.
   This proves the hook is on the value the sweep consumes, and it means **emulated iteration counts are the real
   ones** for every format implemented the same way (store narrow, compute fp64).
3. Each rounded variable reports a nonzero relative change (`delta`); a zero where rounding was asked stops the
   search (`RoundingNoOp`, the HDCG rule).
4. `check_ex12_eul.sh` green; the default build bitwise unchanged (rounding off costs one branch per store).

### Stage B -- the L2 search on the Mac (Parsimony, 1 session, Opus)

`kintora_eval.py` on the co-process evaluator (P3), the grouped search (P6), the joint grid over `dinv` x `coup`
(P8).  L2 v12 and L2 v48, Kokkos Serial or OpenMP.  Also from the driver: the condition numbers of the 24 x 24 blocks
per (mu row, dt), the predictor Anzt et al. use to choose block-Jacobi storage per block.  Exit: the L2 tables of the
paper's Sec. 5 from `results/*.json` by `make_tables.py`; per-quantity knees at both slacks 0 and 0.25.

### Fable session -- the prediction (one session, any time before Stage B; brief: `FABLE-BRIEF.md`)

Question: what sets the bits a block-inverse smoother's stored data needs inside a V-cycle?  Run before Stage B,
so its predictions are on record before the data that test them; inputs are the solver structure, KINTORA's numpy
oracle at L2 and the measured contraction factors, not the ladders.  A perturbation argument on the error
propagation operator should predict the knee per quantity from the block conditioning and the smoothing factor, and
predict where the linearized blame model (P7) fails.  This is a derivation whose structure is expensive to get
wrong, which is Mark's criterion for Fable.  If the derivation shows one quantity sets the knee independent of mu,
dt and v resolution, the session says so and stops early; the rest is Opus work.

### Stage C -- the A100 search (Perlmutter, Mark's m1489_g node, 1-2 sessions, Opus)

L4 v12 + L4 v48 for the search; the winner confirmed at L5 and L6.  Per-level variables named by depth below the
finest, so a configuration found at L4 applies at L5; report whether a level's requirement follows its depth or its
mesh (the streaming Courant number grows 7/16/32 from L2 to L4).  `-kin_mem_view` at L5 and L6 per array.  Runner
rules below.

### Stage D -- real storage, time and memory (KINTORA + Parsimony, 1-2 sessions)

For the formats at the knees: real storage of `dinv` and `coup` (the sweep's `DT` template already takes `float`;
add `Kokkos::Experimental::bhalf_t` / `half_t`), so the stored bytes actually drop.  Measure per V-cycle time
(nsys, after session 8's sweep work), device memory per array, and the largest L that fits one 40 GB A100.  Compare
emulated and real iteration counts (gate 2 predicts equality).  Variant: per-block adaptive precision chosen by
block condition number (Anzt, Dongarra, Flegar, Higham, Quintana-Orti 2019) against the searched per-quantity
uniform format -- a rule against a search, on the same blocks.

### Stage E (optional) -- arithmetic precision by verificarlo/VPREC (2-3 sessions)

Storage rounding answers "how few bits stored"; verificarlo's VPREC backend answers "how few bits computed" (LLVM
instrumentation of every FP operation).  It cannot see CUDA device code, so it runs on the Kokkos **Serial** build:
libkin + ex12 rebuilt with `verificarlo-c++` against an unmodified fp64 PETSc.  Kokkos inlines kernel helpers into
lambdas, so function-level VPREC configuration needs `noinline` on `EulSweep24Load`/`R`/`Y`, `EulMFNodeA`/`B`,
`EulTeamInvert`, ... during exploration.  ~10-100x slowdown: L2 only.  The first arithmetic question is
`EulTeamInvert` (the 24 x 24 Gauss-Jordan, 6 ms per level per row, session-8 item 2): can it run in fp32?  Its MCA
mode also gives an independent significant-digit estimate for the conservation ledgers.  This would be Parsimony's
first non-numpy evaluator plugin.

## Library track (Parsimony)

Each item is a tested increment in `src/parsimony/`, landed in the stage that first needs it.  The test for "does
this belong in the library" is that HDCG and KINTORA both use it.

| | item | why | stage |
|---|---|---|---|
| P0 | **fp8 e4m3 as OCP E4M3FN**: max 448, no infinity, overflow to NaN -- **DONE 2026-10-10** | it was IEEE-style (max 240, overflow to inf), so values in (240, 448] read as overflow; fixed in `formats.py` and `round_bits.h` (version 2), tested against all 256 encodings decoded from the OCP spec.  The Parsimony paper already quoted 448 | 0 |
| P1 | `parsimony.native`: `round_bits.h` moved from `casestudies/hdcg/`, `get_include()` | both C++ drivers include it; one copy, one agreement test against `round_to` | 0 |
| P2 | `NativeEvaluator` base, extracted from `grid_eval.py` | the run-a-binary loop: JSON contract, memo cache keyed by config, JSONL probe log, positive verification, `RoundingNoOp`, `recalibrate` for slack sweeps | 0 |
| P3 | `CoprocessEvaluator`: the driver stays resident, one config per stdin line, one JSON result per stdout line; several drivers per evaluator (v12 + v48) | set-up is paid once.  L4: ~1 min of mesh and geometry set-up per process against 0.65 s of set-up + solve for 9 cases | B |
| P4 | `RateBudget` criterion (contraction factor), true-residual guard | 2-4 outer its quantize the iteration budget | 0 |
| P5 | weighted `ConfigSpace` (bytes per variable) and a bytes report | the objective is bytes, not summed format widths (`total_bits`) | 0 |
| P6 | grouped variables (quantity x level) and a group-then-refine search | ddmin over 6 quantities, then per level inside the quantities that lowered; hierarchical delta debugging (Misherghi and Su 2006) | B |
| P7 | linearized blame: one perturbed run per variable gives a sensitivity c_v; predicted knee k_v = log2(c_v / tol) and a joint allocation by equal marginal error per byte | the Blame Analysis idea at the level of stored quantities instead of instructions; ddmin checks the prediction | B |
| P8 | joint grid over two variables (dev-plan B1) | prefilter floors are not sound bounds when errors interact (finding 1) | B |
| P9 | `make_tables.py` pattern: paper tables generated from `results/*.json` | every number in the paper traces to a run | B |

## Paper track

`paper/kintora-precision.tex` (pdflatex + bibtex, `make` in `paper/` builds both papers), following `~/Codes/plasma-kinetics/paper/house-style.md`.
Everything not yet earned sits in a `\pending` box.

| section | content | earned by |
|---|---|---|
| 1-2 | introduction, Precimonious, Blame Analysis, Parsimony | written |
| 3 | the solver and its stored data (the table above) | written from D230; memory column after Stage C |
| 4 | method: variables, the rule, hooks and their gates, observables, cases, searches | written; gates after Stage A |
| 5 | bits: uniform floor, per-quantity knees, mixed configuration, v12 vs v48, L4 -> L6 | Stages B, C |
| 6 | what the bits buy: memory, time, real formats, per-block rule vs search | Stage D |
| 7 | the workflow and what moved into Parsimony | track P |
| 8 | limitations: storage vs arithmetic (Stage E), one GPU, the 2X+1V solve of the axisymmetric code | written |

## How the two projects inform each other

The study is not only a consumer of KINTORA.  What it measures is a set of KINTORA design decisions, and each
one goes back as a proposed change with its evidence.

**Parsimony -> KINTORA**

| what | KINTORA decision it informs | when |
|---|---|---|
| bits per stored quantity, with the iteration cost of one format lower | storage defaults (`-eul_mg_sweep_fp32` was the first; C and 16-bit D^-1 are next) | Stages B, C |
| bytes per array, set-up scratch vs solve data | memory fixes (`d_pbinv`, coarse `d_Minv`, already found) and the largest L per GPU, so the mu rows per GPU (`-nrow_blocks`) | now, Stage C |
| which arrays dominate the bytes the sweep reads | whether the sweep-parallelism work (session 8 item 1) pays: if `dinv` and `coup` go to 16 bits, a bandwidth-bound sweep gains in proportion | Stage C, before D |
| the contraction factor's sensitivity to `dinv` rounding, by mu, dt and v resolution | **the smoother's margin**: a controlled perturbation of the smoother shows where it is close to failing (large mu, v48), which is the question behind session 8 item 7 (the coupling between v segments) | Stage B |
| a contraction-factor check | a gate in `check_ex12_eul.sh` finer than "<= 4 its": it registers a smoother regression before it costs an iteration | Stage B |
| fp32 arithmetic in the 24 x 24 inversion (Stage E, or real fp32 in the rewrite) | session 8 item 2 (PBSetUp24, 6 ms per level per row) | Stage E |
| the precision table and the method | the Eulerian solver paper's claim that fp32 inverses cost nothing, on a measured basis | Stage C |

**KINTORA -> Parsimony**: the requirements (variables that are arrays x levels, a bytes objective, a rate criterion,
a resident driver) and the test of the library's native layer on a second code.

**The channel.**  `FINDINGS-FOR-KINTORA.md` is the log: each finding with its evidence, the proposed KINTORA action,
and its status (drafted / sent / adopted / rejected, with the ledger entry that records the outcome).  A finding
reaches KINTORA as a short brief Mark relays to the active KINTORA session, written so that session can enter it
as a design-ledger row (date, claim, measured, sources).  Parsimony never edits KINTORA's tree or its ledger.
Every stage's `HANDOFF.md` lists the findings it added.

## Coordination with KINTORA development

- Parsimony sessions never edit `~/Codes/plasma-kinetics`.  KINTORA changes (Stages A, D) are made by sessions run
  from that repository; `DRIVER-CONTRACT.md` is their spec, and they commit under that repository's rules.
- Pin the KINTORA commit and the full option string in every result JSON; a precision result is a property of the
  algorithm (arc length, V(3,3), block size, segment count), not of the kernel's implementation.  Session 8 may
  change the default arc length (item 0): re-run, don't reuse.
- `round_bits.h` reaches KINTORA as a vendored copy with a version line, and Parsimony's agreement test diffs the two
  when both trees are present (decided 2026-10-10, over a build-time dependency on Parsimony: KINTORA builds
  without Parsimony).  The copy states the `PARSIMONY_ROUND_BITS_VERSION` it took.

## Rules (from the KINTORA runs that produced these numbers)

- Pass/fail by outer iterations at fixed rtol plus the fp64 true residual, never error against fp64.
- Include the stiff set; never conclude from v12 alone.
- At least 3 mu rows and 3 dt.
- Turn the existing fp32 off (`-eul_mg_sweep_fp32 0`) whenever the `dinv` hook is under test, so it sees fp64 input.
- Perlmutter: write runner outputs to absolute paths; `make -j8 libkin.a`, not plain `make`; never rebuild a tree a
  queued or running job uses (copy it with `cp -a`); run on Mark's node with `srun --jobid=J --overlap`; never edit a
  running bash script.  `-log_view_gpu_time` for any time measurement; full logs saved, parsed locally.

## Decided (Mark, 2026-10-10)

1. P0: `FP8_E4M3` is OCP E4M3FN -- done (see the library track).
2. Title: the 2X+1V solve.  KINTORA's model is 3X+2V; each system the V-cycle solves is (R, Z, v_par) on one
   poloidal plane at fixed mu.  The paper says both.
3. `round_bits.h` is vendored into KINTORA with its version line; the agreement test diffs the copy.
4. Stage A: a dedicated KINTORA session started from `DRIVER-CONTRACT.md`, after session 8 commits.
5. The paper is `paper/kintora-precision.tex`.
6. The analysis between Stages B and C goes to Fable: `FABLE-BRIEF.md`.  It derives its predictions BEFORE the
   Stage B data exist, so Stage B tests them rather than fits them.

## Where things are

- KINTORA: `src/kin/eul/kokkos/kin_eulop.kokkos.cxx` (kernels; the hook sites), `src/kin/eul/kin_eulmg.cxx` (MG,
  options, `EulMFLinesRichardson`, `EulMFResidual`), `src/kin/gates/kin_gate_eul.cxx` (`EulMGSolveGate`, the
  `-eul_mg_solve` gate with `-eul_mg_rows`, `-eul_mg_dts`, `-eul_mg_cycles`, `-eul_mg_lu`), `scripts/check_ex12_eul.sh`.
- KINTORA results behind this plan: `plans/design-ledger.md` D230-D230(d) (local), Perlmutter `~/eul/node8_*.2101.txt`
  .. `node11_*.2126.txt` (timing sweep), `~/eul/row18_*.eulgpu3.out` (the v48 stall), `~/eul/L6_node.txt`,
  `L6b_node.txt`.
- Parsimony: `casestudies/hdcg/grid_eval.py` and `round_bits.h` (the first native evaluator, to be generalized),
  `casestudies/kintora/DRIVER-CONTRACT.md`, `casestudies/kintora/FABLE-BRIEF.md`, `paper/kintora-precision.tex`.
