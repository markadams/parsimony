# Case study: end-to-end PETSc elasticity solve

Does `time = iterations × cost-per-iteration` predict wall time for a real solver,
rather than for the extracted SpMV kernel of [`../spmv`](../spmv)?

**Platform: the `petsc-claude` fork, not upstream PETSc.** Upstream has no
BAIJ-Kokkos (`MATMPIBAIJKOKKOS` and its implementation directories are absent)
and no Kokkos FE template path, so neither the blocked-GPU arm nor GPU assembly
is reproducible there. That is a real limitation on reproducibility and, read the
other way, the reason the work is worth doing.

Harness is `src/snes/tutorials/ex56k.kokkos.cxx` — 3D linear elasticity with
Kokkos GPU-resident FEM assembly. It is the right harness for two reasons:
`baijkokkos` is assembled **natively** by device block-COO (no `-block_mat_type`
conversion), and it exposes a second kernel class. Assembly is compute-bound
(quadrature and tensor contraction); SpMV is bandwidth-bound. They respond to
reduced precision differently, and that turns out to be the main result.

Measured on one A100-SXM4-40GB, Q2 hexahedra, GAMG + Chebyshev/Jacobi,
`-n_solves 2` with the steady-state solve in its own `HotSolve` stage. Raw logs
in `ex56k-58781480/`.

## Results

Times are the `HotSolve` stage, in milliseconds.

| run | dof | its | displ | Jac assembly | KSPSolve | MatMult |
|---|---|---|---|---|---|---|
| f32 double gpu-scalar | 750k | 31 | 59.9589 | 296.8 | 275.5 | 237.0 |
| f32 double gpu-block  | 750k | 31 | 59.9589 | 288.5 | 386.2 | 352.8 |
| f32 single gpu-scalar | 750k | 31 | 59.9581 | **191.8** | **217.8** | 181.8 |
| f32 single gpu-block  | 750k | 31 | 59.9581 | **174.9** | 462.4 | 434.7 |
| f40 double gpu-scalar | 1.48M | 32 | 59.9541 | 578.4 | 525.3 | 479.8 |
| f40 double gpu-block  | 1.48M | 32 | 59.9541 | 554.6 | 750.1 | 717.1 |
| f40 single gpu-scalar | 1.48M | 32 | 59.9558 | **376.3** | **405.1** | 363.8 |
| f40 single gpu-block  | 1.48M | 32 | 59.9558 | **341.6** | 913.0 | 882.2 |

### fp32 pays, and it pays most in assembly

| | 750k | 1.48M |
|---|---|---|
| assembly, scalar | **1.55×** | **1.54×** |
| assembly, block | **1.65×** | **1.62×** |
| KSPSolve, scalar | 1.26× | 1.30× |
| MatMult, scalar | 1.30× | 1.32× |

This is the first end-to-end precision speedup in the study, and it is consistent
across both mesh sizes and both layouts. Assembly is compute-bound, so fp32 cuts
both the arithmetic and the traffic; the ~1.3× on the scalar solve matches the
byte model of [`../spmv`](../spmv) for AIJ.

**Iteration counts are identical everywhere** — 31 at 750k, 32 at 1.48M —
independent of precision, layout, and whether the Jacobian was assembled on the
host or the device. Displacements agree to 5–6 digits. So the tier-1 result holds
on a real elasticity solver: fp32 storage costs no iterations.

Composing the tiers on the scalar arm at 1.48M dof: fp32 costs 0 iterations and
buys 1.54× on assembly and 1.30× on the solve, for **~1.4× overall at the same
answer**.

### The block layout is slower, and fp32 makes it worse

This contradicts both the byte model and the microbenchmark, so it is stated
plainly rather than buried.

| MatMult, per call | double | single | double→single |
|---|---|---|---|
| `seqaijkokkos` | 547 µs | 420 µs | **1.30× faster** |
| `seqbaijkokkos` | 815 µs | 1004 µs | **0.81× — slower** |

At 750k dof the block MatMult is 1.49× slower than scalar in double and 2.39×
slower in single; achieved rate goes 190→248 GFlop/s for scalar (improving) and
128→104 for block (degrading).

The byte model says block should move ~29% *fewer* bytes at `bs=3`, and the
standalone Kokkos kernel in [`../spmv`](../spmv) measured block 1.64× *faster*
than scalar. PETSc's `SEQBAIJKOKKOS` MatMult does the opposite on both counts.

Checked, and none of these explain it:

- The operator really is `seqbaijkokkos`, confirmed by `-ksp_view` (`type:
  seqbaijkokkos`, `bs=3`) against `seqaijkokkos` for the scalar arm.
- `MatConvert` appears only in the block run but costs 10 ms, against a 116 ms gap.
- MatMult call counts are identical (433), the hierarchy is identical, and the
  converged answer is identical, so this is not a different amount of work.

This looks like a real regression in the BAIJ-Kokkos MatMult, worst in single
precision. It is a PETSc implementation matter rather than a finding about
precision tuning, and it does not affect the scalar-arm conclusions above.

### GPU versus CPU assembly

CPU-assembly control at 750k dof: assembly 301 s against 0.30 s on the device.
That ratio is **not** a fair architectural comparison — it is one CPU core
(`srun -n 1`, unthreaded PETSc) against a full A100, and should not be quoted as
a GPU speedup. Its purpose here is a correctness control, and it serves: the CPU
arm reproduces the same displacement (59.9589) and the same 31 iterations.

Worth noting for the study's framing: with CPU assembly, assembly was ~12× the
solve; on the device the two are comparable (297 ms against 276 ms). Both halves
of the run matter once assembly is on the GPU.

## Method notes, learned the hard way

**`-log_view_gpu_time` is mandatory on this path.** Without it PETSc's async GPU
MatMult logs only launch cost (0.4 µs/call, *falling* as the problem grows) and
the next synchronisation point — `VecTDot` — absorbs the real work, reporting
91–95% of `KSPSolve`. The giveaway was a dot product at 1.48M dof implying
10.6 GB/s, half a percent of peak. With the flag, MatMult is 86–97% of
`KSPSolve`, as it should be. Every per-event number from a plain `-log_view` run
on this path was discarded.

**Do not price a multigrid MatMult with one nnz/row.** An effective-bandwidth
figure computed that way came out at 2489 GB/s, above A100 peak: the 433 MatMult
calls span the whole GAMG hierarchy, not the fine grid. Only ratios between runs
with an identical hierarchy are used here.

**Read the example's `/*TEST*/` block before inventing arguments.** The block path
on `ex56.c` is `-dm_mat_type mpiaij -block_mat_type mpibaijkokkos`, not
`-dm_mat_type baijkokkos`; and `-dm_mat_type` alone is not enough there, because
the vectors are created from the DM before the matrix exists.

## Files

- `run_ex56k.sh`, `ex56k-58781480/` — this study
- `run_scale.sh`, `scale-*/` — the published ex56 test at four mesh sizes, which
  diagnosed the async attribution artifact
- `run_e2e.sh`, `run_prec.sh` and their logs — earlier ex56 sweeps; iteration
  counts are good, per-event times predate `-log_view_gpu_time` and are not used
- `ex56k-{single,double}` are built on Perlmutter from
  `arch-perlmutter-opt-gcc-kokkos-cuda{-single,}` and are not in the repo
