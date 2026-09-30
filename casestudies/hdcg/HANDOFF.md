# Handoff: Grid solver precision case study

Written 2026-09-30 for the next session. Read this, then `NOTES.md` (the full
record: build recipe, patches, ruled-out table, results, production's
precision design). `PLAN.md` is the original HDCG plan and is partly
superseded — where they disagree, this file and NOTES.md win.

## Where we are

- **Grid `develop 865c338d` is built** on Perlmutter with CUDA, and the
  rounding machinery runs on GPUs against real Grid data.
- **First measurement done**: the coarse Dirac stencil, rounded in storage,
  leaves CG at 5 iterations at every format down to 8 bits, while the
  attainable residual tracks 2^-k. fp32 and bf16 reproduce the 24- and 8-bit
  virtual widths bit for bit. Free field, so preliminary.
- **The original HDCG driver is blocked** by a Grid-side problem:
  `MultiGeneralCoarsenedMatrix` (in `multigrid/deprecated/`) segfaults on
  construction at every setting tried. It is Q1 in `share/QUESTIONS-FOR-PETER.md`.
  Do not spend more time debugging it.

## What is next

A case study on **a production Grid solver: the mat-vec and the smoother,
against accuracy and iterations**. Mark met Joe Pusztay on 2026-10-01 to pin
down what production is.

**First action: ask Mark what Joe said.** The driver cannot be designed until
these are answered:

1. Which solver and preconditioner is production, and which driver on develop
   builds it.
2. Is the preconditioner fp32 arithmetic (the `MrhsMixedPrecPreconditioner`
   seam) or all fp64? This decides the emulation, see below.
3. Production volume, Ls, mass, M5, solver tolerance.
4. **Three gauge configurations** from one ensemble, decorrelated in Monte
   Carlo time, and their Perlmutter paths.
5. File format. NERSC format reads today; ILDG `.lime` needs Grid rebuilt
   `--with-lime`.

## Design decisions already settled

**The stored operator data is the gauge links.** The fine Dirac operator is
matrix-free: Grid never assembles it. γ-matrices are fixed spin projections and
the Möbius fifth dimension has scalar coefficients. So "store the matrix in low
precision" means: compute the (doubled) links in full precision, store a
low-precision copy. Rounding the links *before* building the operator is exact
to do, because adjoint, negation and the antiperiodic sign all preserve
representability.

**Vector precision must be a variable too.** For domain-wall fermions one 4D
link serves all Ls slices, so links are a small share of the mat-vec traffic —
roughly 72 B of links against 960 B of spinors per 5D site at Ls = 16. That is
an estimate from the kernel structure; measure it before relying on it. If it
holds, bf16 links buy only a few percent and the fermion vectors are what pays.

**Match production's precision boundary.** Verified from source 2026-09-30:
`MrhsMixedPrecPreconditioner` (`multigrid/MrhsMultiGrid.h:267`) runs the
*whole V-cycle* in fp32, data and arithmetic, and the outer solve stays fp64.
`innerProduct` forms each site's product in the field's precision and sums
across sites in double. So our fp64 storage-only emulation is *optimistic*
against production. The faithful design:

- outer solve on the exact fp64 operator;
- preconditioner on Grid's real single-precision types (real fp32 arithmetic,
  Grid's own reductions);
- round the fp32 link copy, and the fp32 vectors, below 24 bits.

That reproduces production exactly at fp32 and extends it to bf16/fp16 storage.
If Joe says production is all fp64, the storage-only model already matches and
this is unnecessary.

**Sweep the joint space, not one variable at a time.** The solo-sensitivity
prefilter is a heuristic, not a bound: a coarse format downstream can absorb
upstream error. With two or three variables the full cartesian product is
affordable and shows the interaction directly.

**Many points per process.** Setup was 99.95% of per-point cost last time
(53.6 s setup, 0.017 s solve). Build the operators once, keep an exact copy,
then loop: restore from the copy → round → solve → append a record. Rounding
is in place, which is why it needs the restore.

**Three configurations × two RHS cases.** Configurations vary the operator's
spectrum; RHS cases only vary the source.

## Work list, in order

1. Get the answers above from Mark.
2. Add a float variant of `RoundField` in `round_field.h` (it rejects float
   fields on purpose today; keep the by-name type check). Extend
   `tests/test_hdcg_round.py` to cover rounding float inputs below 24 bits.
3. Add an exhaustive-product search to parsimony (~20 lines, next to
   `src/parsimony/searches/uniform.py`), with a test against a planted
   two-variable interaction.
4. Write the driver for the production solver: exact fp64 outer operator,
   fp32 preconditioner operator built from a `precisionChange`'d gauge field,
   `--round-links`, `--round-vectors` (names up to you), in-process sweep loop,
   JSON record per point.
5. Put `ranks`, `mpi` and the Grid commit into every JSON record — last time's
   records lacked them.
6. Drive the new driver through `grid_eval.py` rather than a shell loop — the
   evaluator is tested but has never run against Grid for real.
7. Verify the v53/fp32 point reproduces the unmodified production solver's
   iteration count exactly before running any sweep.
8. Sweep: 3 configurations × 2 cases × the product ladder.

## Where things are

| what | where |
|---|---|
| study record | `casestudies/hdcg/NOTES.md` |
| rounding kernel, tested vs `formats.round_to` | `casestudies/hdcg/round_bits.h` |
| Grid lattice wrapper | `casestudies/hdcg/round_field.h` |
| working driver (coarse stencil) | `casestudies/hdcg/Test_coarse_bits.cc` |
| blocked HDCG driver | `casestudies/hdcg/Test_hdcg_parsimony.cc` |
| parsimony evaluator for Grid | `casestudies/hdcg/grid_eval.py` — tested only against a stand-in driver; it has never driven Grid on the machine (the published ladder came from the shell loop in `jobs/run_sweep.slurm`) |
| job and build scripts | `casestudies/hdcg/jobs/` (copied from Perlmutter scratch) |
| results, 24 JSON + log | `casestudies/hdcg/results/` |
| questions for Peter, slides, walkthrough | `casestudies/hdcg/share/` |
| code tour of every Grid change (interactive) | `share/grid-precision-hooks.html`, https://claude.ai/code/artifact/c0cae33e-6cba-453a-8cc2-ae46e4fa972c — regenerate with `python3 share/src/gen_hooks.py` after editing any file it quotes |
| our two Grid patches (apply to `865c338d`) | `casestudies/hdcg/grid-patches/` |
| Grid source + build | `/pscratch/sd/m/madams/Grid`, `/pscratch/sd/m/madams/Grid/build` — scratch is purged when idle; rebuild from `NOTES.md` + `grid-patches/` |
| job scripts, run outputs | `/pscratch/sd/m/madams/hdcg/` |

## How to build and run

Drivers are compiled by hand, because they have no `Makefile.am` entry.
Copy the source and both headers into Grid's `tests/debug/` first:

```bash
scp -i ~/.ssh/nersc Test_coarse_bits.cc round_field.h round_bits.h \
    madams@perlmutter-p1.nersc.gov:/pscratch/sd/m/madams/Grid/tests/debug/
# on Perlmutter; the argument is nbasis
bash /pscratch/sd/m/madams/hdcg/build_bits.sh 20     # -> Test_coarse_bits_nb20, ~7 min
```

`build_driver.sh` is the same for `Test_hdcg_parsimony`. Copy either as the
template for the new driver. Jobs: `-A m499_g -C gpu -q debug`, one node, four
ranks, the `select_gpu` wrapper and `MPICH_GPU_SUPPORT_ENABLED=1` — see
`run_sweep.slurm`. Use `srun`, not `mpiexec`.

## Gotchas that each cost hours

- **`srun` exit 0 does not mean success** once `--debug-signals` is on. Grid
  catches SIGSEGV, prints a backtrace and exits clean. Grep for
  `Caught signal`.
- **Never filter output inside a job script.** Last sweep piped `srun` through
  `grep` and lost the MPI banner. Save full logs, parse afterwards.
- `pgrep nvcc` returning 0 does not mean a build died; use the log's mtime.
- `autoView` and `accelerator_for` are macros: they can't be
  namespace-qualified, and a function outside namespace Grid needs
  `using namespace Grid;` for them to expand.
- Grid allocates `_Adag` only for a *non*-Hermitian coarse operator.
- The NERSC certificate lasts 24 hours: `sshproxy.sh -u madams`. `sacct`
  sometimes fails on login nodes with munge errors; that is NERSC-side.

## Tree state at handoff

Committed on `main`, not pushed: `5db5cd3` (case study: code, tests, results,
patches), `ffdd50a` (material for Peter), `86681b7` (code tour, job scripts).
104 tests pass. `.vscode/` is Mark's editor config and stays untracked.
