# Parsimony development plan

Written 2026-10-03. Audience: Mark and later Claude sessions. Covers now through
January 2028. Sub-plans hold the detail; this file holds the order,
the dependencies and the reasons.

State today: 104 tests pass. The paper (`paper/parsimony.tex`) is complete, with
PETSc results. Grid is built on Perlmutter, and one coarse-stencil ladder is
measured. The repo has been public since 2026-10-03.

**Grid is no longer part of this plan (2026-10-06).** Grid work continues in this
repo as GPU performance analysis on Frontier, not precision tuning: see
`grid/FRONTIER-SETUP.md`. The Grid precision study stays recorded in `casestudies/hdcg/`.

## Workstreams

| | Workstream | Sub-plan | Waits on |
|---|---|---|---|
| A | Paper: arXiv first (venue later); co-authors may join | `paper/REVIEW-PROMPT.md` | nothing |
| B | Core tool: searches and evaluators | this file | nothing |
| D | PETSc case study at scale | `casestudies/e2e/README.md` | node-hours on existing NERSC projects |
| E | Parsimony–PETSc integration, ending in an upstream MR | `casestudies/petsc/PLAN.md` | B2; MatScalar Stage B for its stage 3 |

## B. Core tool

What the case studies need, in the order they need it:

1. **Joint grid search** (~20 lines + tests). An exhaustive Cartesian sweep over
   a few variables, such as operator storage x smoother precision in PETSc. Prefilter
   floors are not sound bounds when errors interact (finding 1). Every existing
   search moves one variable at a time or bisects uniformly.
2. **Restore-and-repeat evaluator base.** The "set up once, keep an exact copy,
   then restore, round, solve and record each point" loop, written once and shared
   by `grid_eval.py` and the PETSc evaluator. Setup was 99.95% of the cost of each
   point in the Grid run.
3. **GPU-timed evaluator.** Cost per iteration measured on the device, so
   `time = iterations x cost per iteration` comes from data, not the byte model.
   The search still runs on iterations, because modeled time is non-monotone; time
   is applied afterwards as a Pareto curve over slack.
4. **Coarse-grained blame.** Attribution per library operation (a MatMult, a
   level of the V-cycle) rather than per instruction. This is the idea that makes
   the method affordable on a real HPC library, and the PETSc solver-tree walk
   (E, stage 0) is its first instance.
5. Lower priority: a blame-style prefilter (cuts ddmin's O(n^2) probes), NAS `cg`
   validation against the published Table 4/5, and a PyPI name (`parsimony` is
   taken; PEP 541 or a fallback name, decided at first publish).

## Where MPI enters, and where it does not

Parsimony's search does not need to know about MPI. An evaluator runs `srun`, and
rounding is elementwise, so it does not depend on the decomposition. MPI matters
in four places:

1. **The payoff depends on local size.** Reduced storage speeds up the
   bandwidth-bound local work and leaves latency alone. That includes the message
   latency of halo exchanges and allreduces, GPU kernel-launch overhead, and AMG
   coarse levels. In the strong-scaling limit those dominate, and precision buys
   close to nothing. So D measures **weak** scaling at production size per GPU,
   and every speedup is quoted with its local size.
2. **PETSc communication is unchanged by matrix-only reduction.** Halos carry
   vector values. With fp32 Mat storage and fp64 vectors (MatScalar Stage B), the
   bytes sent are unchanged. Reducing communication precision means making
   ghost-vector precision a variable. That is emulatable by rounding ghost values
   after the scatter, but it pays only where halo messages are bandwidth-bound, not
   latency-bound.
3. **Grid already has a hook here.** `WilsonCompressor.h:64` says "Compress
   includes precision change if mpi data is not same". The halo is already
   spin-projected to half spinors. Whether production lowers its precision is a
   question for Peter or Joe, and it is a candidate variable for C, since lattice
   QCD is communication-bound across nodes.
4. **Results depend on the decomposition through the setup.** GAMG aggregation
   and Grid's subspace change with the partition. Compare configurations only at
   an identical decomposition and hierarchy, and record ranks and `--mpi` in every
   JSON. The Grid JSON records do not do that yet.

## Sequence

Parsimony has no allocation of its own. GPU runs use Mark's existing NERSC
projects (m1516_g, m4267_g).

| When | Work |
|---|---|
| Oct–Dec 2026 | A: arXiv. B1, B2. E stage 0 (petsc4py on a laptop). |
| Q1 2027 | E stage 1 (A100). B3. |
| Q2 2027 | D: weak scaling 1–64 nodes. Fable break for the PETSc API. |
| Q3 2027 | E stages 2–3 (in-library rounding; real fp32 via MatScalar). B4. Second paper: PETSc at scale. |
| Q4 2027 | Reviewer reruns. An ERCAP request of its own (due about Oct 2027), built on these results. |

## Model use

Opus by default. Fable for two things only: the PETSc precision API design
(E, between stages 1 and 2), and the MatScalar design break already in that plan.

## Risks

- **Emulation is optimistic.** It models storage, not arithmetic. In PETSc, E stage 3
  measures the gap directly.
- **The single-GPU blocked-format result does not predict scale.** Check Mark's
  BAIJ-Kokkos paper before calling any blocked-format number an anomaly.

## Settled (2026-10-03)

- **Paper**: arXiv is the placeholder venue. No co-authors yet; more people will
  probably join.
- **PETSc**: the target is an upstream MR. The `petsc-claude` fork is for a good
  BAIJ matrix for papers, and for anything that should not go upstream.
- **Grid**: dropped from this plan 2026-10-06; see `grid/FRONTIER-SETUP.md`.
- **Allocation**: no Parsimony ERCAP this year (2026-10-06).

## Open questions for Mark

- Parsimony gets its own repository: when, and does `grid/` move with it?
