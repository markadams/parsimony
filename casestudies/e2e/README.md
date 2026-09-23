# Case study: end-to-end PETSc solve (in progress)

Checks whether `time = iterations × cost-per-iteration` predicts wall time for a
real PETSc solve rather than for the extracted SpMV kernel of
[`../spmv`](../spmv). **Not finished** — the precision question is still open.

Arguments are taken verbatim from the `ex56` kokkos "hot" testset (suffixes
`hot_scalar` / `hot_block`), mesh size aside. Two things that matters for:

- `-n_solves 2` runs a second solve reusing the GAMG hierarchy, logged in its own
  "Hot Solve" stage. Steady-state cost is read from that stage; the first solve
  carries setup.
- DMPlex assembly of the *Kokkos BAIJ* structure goes through
  `-block_mat_type mpibaijkokkos` in that testset rather than
  `-dm_mat_type baijkokkos`.

## Established: iteration counts are precision- and layout-independent

At 311k dof, all four combinations of {double, single} × {scalar, block} converge
in **232** iterations (Jacobi) and **23** (GAMG), with the final displacement
agreeing to 7 digits. Block math equals scalar math, and fp32 storage costs no
iterations — the tier-1 result from [`../newton`](../newton), reproduced on a
real elasticity solver.

## Established: the per-event profile is misattributed, and the solve is healthy

The first run appeared to show `VecTDot` consuming 91–95% of `KSPSolve` and
`MatMult` 1–2%, which would be pathological. It is an artifact. Scaling the
published test across four meshes settles it:

| per-call µs | 89k dof | 311k | 750k | 1.48M |
|---|---|---|---|---|
| MatMult | 0.9 | 0.7 | 0.5 | **0.4** |
| VecTDot | 187 | 474 | 1195 | **2241** |

`MatMult` at 0.4 µs/call is a kernel *launch*: PETSc's GPU MatMult is
asynchronous, so the logged time excludes the work. `VecTDot` is the
synchronisation point and absorbs it. The giveaway is that a dot product at
1.48M dof moves 23.7 MB, so 2241 µs implies **10.6 GB/s — 0.5% of A100 peak**,
impossible for a real reduction and exactly what absorbing the SpMVs looks like.

`CpuToGpu` and `GpuToCpu` counts are **zero** throughout the Hot Solve stage, so
nothing round-trips to the host; that suspicion was wrong.

The aggregate is fine: **12.3 ms per iteration at 1.48M dof** for a GAMG V-cycle,
scaling *sublinearly* (16.5× the dof for 10.7× the time). Nothing is broken.

**Consequence for this study:** only `KSPSolve` wall time — a synchronisation-to-
synchronisation measure — is trustworthy without `-log_view_gpu_time`. Any
per-event number from a plain `-log_view` run on this path should be discarded.

## Open: fp32 storage bought nothing end-to-end

`KSPSolve` wall time was identical between the single- and double-precision
builds (1.00× in all four configurations at 311k dof). That is a real
measurement, not an attribution artifact.

It is also unexplained. At 311k dof the Jacobi solve runs 0.53 ms/iteration,
while the SpMV alone should cost ~0.37 ms at fp64 bandwidth — so the solve looks
operator-bound, and the byte model of [`../spmv`](../spmv) predicts ~1.4× for
scalar fp32. Job in flight repeats the comparison at 750k and 1.48M dof with
`-log_view_gpu_time`, which forces per-event synchronisation, and pairs each run
with a plain one so the profiling overhead is visible.

Until that resolves, **no claim from this directory belongs in the paper.**

## Files

- `run_scale.sh`, `scale-*/` — the published test at four mesh sizes, plain `-log_view`
- `run_prec.sh` — single vs double at 750k/1.48M with and without `-log_view_gpu_time`
- `run_e2e.sh`, `logs-*/` — the first sweep; iteration counts are good, per-event times are not
- `ex56-single`, `ex56-double` are built on Perlmutter from
  `arch-perlmutter-opt-gcc-kokkos-cuda{-single,}` and are not in the repo
