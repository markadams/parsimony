# Parsimony–PETSc integration: development plan

Written 2026-10-03. Audience: Mark and later Claude sessions (Opus throughout;
one Fable break, marked). Every session ends at a checkpoint: clean tree, tests
green, a `HANDOFF.md` here with tree state and the exact next step.

## Decision

Parsimony stays outside PETSc and drives it through **petsc4py**. PETSc gains only
per-object precision controls, which are useful without Parsimony. Neither code
depends on the other.

- **Variable names come from the solver tree.** After `KSPSetUp`, walk PCMG levels
  and name each object by its options prefix (`mg_levels_2_` operator, level-2
  interpolation, `mg_coarse_` solve). That walk is `discover()`.
- **The observable comes from PETSc.** `KSPGetIterationNumber`, `KSPGetConvergedReason`,
  and a true residual `‖b − A x‖` computed with the untouched fp64 operator.
  That feeds the existing `IterationBudget` criterion: case keys
  `case{i}.converged` and `case{i}.iterations`.
- **Many points per process**, the Grid design (`../hdcg/grid_eval.py`). Set up
  once, keep an exact `MatDuplicate(COPY_VALUES)` of every rounded object, then
  for each point: restore (`MatCopy`, same pattern), round, solve, record.

Not doing: a configure and rebuild per configuration (`PetscReal` is a
configure-time typedef, and one GAMG hierarchy needs both precisions); a global
`MatScalar = float` (rejected in the MatScalar plan); instruction-level
instrumentation of PETSc (Blame Analysis measured 50× overhead on NAS).

## Rules (each cost a run before)

- Matrices with generic entries only. A constant-coefficient Laplacian is exact
  in fp8 (finding 3). Use ex56 elasticity.
- At least 3 right-hand sides (`cases=`), or the search quantizes onto a lucky
  output (finding 2).
- Pass/fail from iterations plus the independent fp64 true residual, never error
  against the fp64 run (finding 4).
- Only preconditioner inputs are rounded. The outer `Amat` stays fp64, so the true
  residual needs no special build.
- On GPU: `-log_view_gpu_time` always. Full logs saved, parsed locally, never
  filtered in the job script. Verify positively (look for the converged line).

## Stage 0 — petsc4py prototype, no PETSc changes (Opus, 1–2 sessions, laptop)

Files: `casestudies/petsc/petsc_eval.py`, `casestudies/petsc/run_ex56.py`, tests in
`tests/test_petsc_eval.py` (skip if petsc4py is not importable).

1. Problem data: run `snes/tutorials/ex56` (petsc-claude, `arch-macosx-gnu-O`, which
   has petsc4py) with `-ksp_view_mat binary` / `-ksp_view_rhs binary`. Also save
   the 6 rigid-body near-null-space vectors; GAMG needs them and the matrix file
   does not carry them.
2. `PetscEvaluator(Caps(needs_rebuild=False, sees_gpu=False))`: load A, b and the
   null space; `KSPSetOperators(A, P)` with `P` a copy of `A`; GAMG +
   Chebyshev/Jacobi; `KSPSetUp`; walk the levels.
3. Rounding by `getValuesCSR` → `formats.round_to` → `setValuesCSR`, host AIJ only.
   Slow and fine for a prototype.
4. Variables, first cut: each level operator `A_l` for l ≥ 1, each interpolation
   `P_l`, the fine-level preconditioning matrix `P`. Two modes for `P_l`, the same
   distinction as Grid's subspace/stencil: round **before** the Galerkin product
   (stays consistent) or **after** it (operator and transfer disagree).

**Verify before trusting the first number:**
- Changing a level matrix in place bumps its object state. Find out what that
  makes PCMG redo on the next `KSPSolve` (smoother setup, Chebyshev eigenvalue
  bounds). The bounds must not be re-estimated silently between points: either
  freeze them or record that they moved.
- Which matrix the fine-level Chebyshev smoother applies: `Amat` or `Pmat`.
  Only `Pmat` is rounded.

Exit: `v53` everywhere reproduces the baseline iteration count exactly, on every
case. A per-level bit ladder for GAMG on ex56 at laptop size. Tests pass
(`.venv/bin/pytest`). Checkpoint: `HANDOFF.md` with the ladder table.

## Stage 1 — GPU and time (Opus, 1–2 sessions, Perlmutter)

1. Build petsc4py on `arch-perlmutter-opt-gcc-kokkos-cuda` (check whether it is
   already there first).
2. Same evaluator on `aijkokkos`. Rounding still goes through host CSR.
   Acceptable while setup dominates; measure and record the cost per point.
3. `Caps(sees_gpu=True, measures_time=True)`: time per point from
   `-log_view_gpu_time` events (KSPSolve, MatMult per level), parsed locally.
4. Whole search inside one `salloc`, as in the Grid study.

Exit: the Stage 0 ladder reproduced on A100 (same iteration counts at each rung),
with GPU time per point. Checkpoint: `HANDOFF.md` with the A100 table.

## Break: Fable, one short design session

Inputs: Stage 0–1 results, the MatScalar plan, the PETSc API that Stage 2 would
add. Question: what per-object precision API PETSc should expose, and whether
emulated rounding belongs upstream at all or stays in petsc4py and the fork. Getting
an API's structure wrong is expensive to undo after an MR, which is why this
gets Fable.

## Stage 2 — in-library rounding (Opus, 1–2 sessions, petsc-claude)

The target is an upstream PETSc MR; the Fable break settles its shape. Anything
that should not go upstream stays in the `petsc-claude` fork.

1. A `Mat` operation that rounds stored values to k significand bits in place,
   with an option on the object's prefix. Host AIJ/BAIJ, and device AIJ-/BAIJ-Kokkos
   as one `parallel_for` over the value array. Name to be agreed with reviewers.
2. Matching `Vec` operation.
3. `petsc_eval.py` switches to it when present and keeps the petsc4py path as a
   fallback. Results must be identical across the two paths.

Gates: `./configure && make && make check && make checkbadSource && make checkclangformat`
on every commit; Kokkos header-dep `touch` before `make all`. Discuss with the
PETSc maintainers before opening an MR.

## Stage 3 — real storage (after MatScalar Stage B)

Depends on `~/Codes/petsc-claude/plans/matscalar-reduced-precision-plan.md`
Stage B (a per-object fp32 copy of the values, `-mat_baij{cuda,kokkos}_values_single`).

1. The evaluator maps a configuration to real options, rather than rounding
   values in place, for any object at fp32.
2. Compare: emulated fp32 versus real fp32 at the same configuration. Iteration
   counts should agree, because emulation covers storage only. A difference
   measures what fp32 *arithmetic* costs.
3. Time from the real run, so the search can optimize measured time.

Exit: the search's chosen configuration, run with real fp32 storage, measured
against its emulated prediction on A100. This is the result for the paper.

## Open questions for Mark

- Which solvers besides GAMG on ex56: a Newton case (Bratu has a measured ladder
  in `../newton`) or an application code?
- Should `PETSc` appear as an optional dependency in `pyproject.toml`, or stay a
  case-study import only?
