# HDCG precision study: plan

Written 2026-09-27. Audience: Mark and later Claude sessions (Opus throughout;
one Fable pass at the end, marked). Goal: the storage bit-knee of each input
to Grid's HDCG preconditioner, measured on Grid itself, in about five working
days, with no code in Peter Boyle's live paths and no involvement of Matt or Joe
beyond one data file.

## What is measured, and what is not

Parsimony's semantics (see `casestudies/newton/DESIGN.md`): **storage** rounding
of a variable to a virtual width `v<k>` (float64 rounded to k significand bits,
full exponent range, round-to-nearest-even, `formats.round_to`), arithmetic in
fp64 throughout. The observable is **outer HDCG iterations at fixed tolerance**
(`IterationBudget`, slack swept 0 / 0.1 / 0.25 for a Pareto curve), judged by an
**independent fp64 true residual** of the fine operator after the solve. Since
only preconditioner inputs are rounded, the fine operator is untouched and the
true residual needs no special build.

Not measured: reduced *arithmetic* (Peter's ComplexF coarse space, 16F/BF16
compute types). Say so in every write-up. The result complements his
fp32-vs-fp64 comparison with a per-variable knee.

## Driver and variables

Base: `tests/debug/Test_general_coarse_hdcg.cc` on Grid `develop` `865c338d`
(deprecated, ComplexD-only classes; stable; nobody is editing it). Ls=16,
nbasis=40, Block 4^4, nrhs=8, Mobius mass 0.01, `TwoLevelCGmrhs(1e-8, 500)`.
Lattice from `--grid 16.16.16.32`; gauge `ckpoint_lat.4000` (NERSC format).

Copy it to `casestudies/hdcg/Test_hdcg_parsimony.cc` (built in Grid's
`tests/debug/`, so `scripts/filelist` picks it up) and add command-line knobs so
ONE binary serves every configuration, no rebuild per evaluation:

| variable | where rounded | access |
|---|---|---|
| `subspace` | after `Aggregates.Orthogonalise()`, before `CoarsenOperator` | `Aggregates.subspace[b]` (public) |
| `stencil`  | after the multi-RHS coarse operator is built | `GetMatrix(p,A)` -> round -> `SetMatrix(p,A)` on `BLAS_A[p]` (public) |
| `evec`     | after `IRL.calc`, before `ImportEigenBasis` | local `std::vector<CoarseVector>` |
| `coarse_src` (optional) | coarse RHS entering the coarse solve | only if a hook is cheap; else drop |

Rounding kernel: one templated `RoundField(Lattice<vobj>&, int bits)` using
`autoView` + `accelerator_for`, `frexp`/`ldexp`/`nearbyint` per real component
(ties-to-even, matching `np.round`). Unit-test it against `formats.round_to` on
a few hundred doubles before any run. Knobs: `--round-subspace k`,
`--round-stencil k`, `--round-evec k` (k=53 = off), `--case n` (selects the
source RNG seed; use >= 3 cases, finding 2 in the notes), `--json path` to emit
`{converged, iterations, true_resid, t_setup, t_solve}` per case.

Expected failure modes (to check the data against, not to assume): rounding
`subspace` keeps the two-level method Galerkin-consistent (the coarse operator
is built from the rounded vectors), so it should degrade gently; rounding
`stencil` breaks consistency between the coarse operator and the projector, so
it may show a sharp knee; `evec` only affects the deflation guess, so it should
be the most tolerant.

## Perlmutter build (Day 1, Opus)

Nothing on Perlmutter yet: no Grid tree, no `ckpoint_lat.4000` under $HOME,
/pscratch/sd/m/madams, or cfs m499/m1516/m4267 (checked 2026-09-27).
Do NOT build Grid through petsc-claude (`grid.py` pins the 2024 CPU-only commit
without `multigrid/`).

1. `git clone -b develop https://github.com/paboyle/Grid.git /pscratch/sd/m/madams/Grid`
   (github reachable from login nodes); `./bootstrap.sh` fetches Eigen 3.4.0.
2. gmp: system `/usr/include/gmp.h` + `libgmp.so` are present. mpfr: only the
   runtime lib, no header -> build mpfr 4.x from source into
   `/pscratch/sd/m/madams/sw/mpfr` (5 minutes), or reuse a PETSc
   `--download-mpfr` install if one exists on the machine.
3. Configure from `systems/Perlmutter/config-command`, adapted: default modules
   are `PrgEnv-gnu gcc-native/14 cudatoolkit/13.2 cray-mpich/9.1`; Peter's
   `sourceme.sh` used cudatoolkit/12.0 and his own spack. Try 13.2 first; if
   nvcc/gcc-14 fights Grid's C++17, `module load cudatoolkit/12.4`. Keep
   `--enable-simd=GPU --enable-shm=nvlink --enable-accelerator=cuda
   --disable-unified --disable-fermion-reps --disable-gparity CXX=nvcc
   CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared"`.
   No `--with-lime` needed: we round in memory, not through files.
4. `make -j16` the library, then only `make -C tests/debug Test_general_coarse_hdcg`
   (each test is its own binary; do not build all of `tests/`).
5. Job template: `systems/Perlmutter/dwf4.slurm` (4 GPUs, `select_gpu`
   wrapper, `MPICH_GPU_SUPPORT_ENABLED=1`), with `-A m499_g -q shared` or
   `-q debug`. `srun`, not `mpiexec`.
6. Baseline: run the unmodified driver with `--grid 16.16.16.32 --mpi 1.1.1.4`
   (or 1.1.2.2). Record iterations, timings of subspace / coarsen / Lanczos /
   solve. Save the FULL log; parse locally; verify positively (look for the
   HDCG converged line, not the absence of "error").

Exit: baseline converges; per-phase timings known. If setup (subspace +
Lanczos) dominates, note it: the search cost is setup x evaluations, since the
RNG seeds are fixed and every evaluation recomputes the same subspace before
rounding it. Caching the subspace needs `--with-lime` (c-lime, small); decide on
Day 2 from the numbers, do not pre-build it.

Checkpoint: `casestudies/hdcg/NOTES.md` with the build recipe that worked and
the baseline table.

## Hooks and glue (Day 2, Opus)

- Driver copy with the knobs above; `RoundField` unit test vs numpy.
- `casestudies/hdcg/grid_eval.py`: a Parsimony `Evaluator` (`Caps(needs_rebuild=False,
  sees_gpu=True, measures_time=True, slowdown=...)`) whose `evaluate(config)` runs
  `srun ... Test_hdcg_parsimony --round-<var> <bits> --case i --json out.json`
  for each case inside the current allocation and feeds `IterationBudget`.
  `discover()` returns the three variables and the ladder
  `v53,v40,v30,v24,v16,v11,v8` (plus `fp32`, `bf16`, `fp16` rungs for the
  headline). Run the whole search inside ONE `salloc`/`sbatch` (a Python loop
  calling `srun`), so queue time is paid once.
- Sanity: `v53` everywhere reproduces the baseline iteration count exactly.

## Search runs (Days 3-4, Opus, background jobs)

Prefilter (solo sensitivity per variable, 3 vars x 7 rungs = 21 solves) then
`PrecisionDDMin` from the prefilter floors. Expect 40-80 solves total; at a few
minutes each on 4 A100s this is single-digit node-hours on `m499_g`. Repeat the
winner at three cases and both slacks. Save all logs and `results-hdcg.json`.

## Analysis (Day 5, one Fable pass, no code)

Inputs: NOTES.md, results-hdcg.json, the baseline log. Questions: which knees are
real (versus finding 4: near-tolerance bit-reproduction), whether the `stencil`
knee is the Galerkin-consistency effect or something else, what the knees say
about bf16/fp16 *storage* of the coarse stencil (the one Grid coding item still
open; pitch only if the stencil knee is <= 8 explicit bits), and the two-paragraph
case-study text for `paper/parsimony.tex` with its limitations: 512 coarse sites
(tiny), one 16^3x32 configuration, storage not arithmetic.

## External inputs

Exactly one: `ckpoint_lat.4000` (16^3x32). Ask Joe for his `GRID_LATTICE_DIR`
path on Perlmutter, or the file. A `SU<3>::HotConfiguration` fallback validates
the pipeline only; its knees are not reportable.
