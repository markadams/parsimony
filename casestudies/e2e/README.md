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

### The block layout is slower here — which is the expected single-GPU result

| MatMult, per call | double | single | double→single |
|---|---|---|---|
| `seqaijkokkos` | 547 µs | 420 µs | **1.30× faster** |
| `seqbaijkokkos` | 815 µs | 1004 µs | **0.81× — slower** |

At 750k dof the block MatMult is 1.49× slower than scalar in double and 2.39×
slower in single.

**The fp64 deficit is not an anomaly and not a regression.** Adams, *A Natively
Blocked, Device-Resident Algebraic Multigrid GPU Path in PETSc*, measures the
same thing on the same library and explains it: the blocked format's advantage is
compressed column indices, so it is realized only when there is halo
communication to amortize index traffic against. On one GPU the tuned scalar
kernel wins. That paper's Table 4, at one GPU, gives block÷scalar SpMV ratios of
1.44× (Q1, 76 nnz/row) and **1.62× (Q2, 169 nnz/row)** — and the deficit *grows*
with nnz/row, because the block advantage is proportionally largest in the
index-bound, low-nnz/row regime. Our 1.49× at Q2 on one GPU sits in that range.
At 27 GPUs the same paper measures block **1.42× faster** on SpMV.

The byte accounting also agrees to the digit: that paper counts 12.0 B per scalar
nonzero for CSR against 8.44 B for BAIJ at `bs=3` with fp64 values and int32
indices, for a traffic-bound ceiling of 1.42×; [`../spmv`](../spmv) derived
1.41× independently.

**So a single-GPU measurement of a blocked format says nothing about it at
scale.** Our `../spmv` microbenchmark measured block 1.64× *faster* on one A100,
which now looks like a property of that hand-written kernel pair rather than a
prediction for PETSc's.

**What is new here is the precision behaviour**, since that work measures only
fp64. Going fp64→fp32 makes the blocked kernel 0.81× — slower — while the scalar
kernel gets 1.30× faster. The byte model predicts the opposite: at fp64 the index
is 4 of 12 bytes per scalar nonzero and blocking saves 30%; at fp32 values shrink
to 4 bytes, the index share rises, and blocking should save 45%, lifting the
blocked ceiling from 1.42× to 1.80×. The measurement moving the other way says
the blocked kernel has left the bandwidth-bound regime at four-byte values — the
same transition [`../spmv`](../spmv) found for fp16, arriving one format earlier
because blocking has already removed most of the traffic.

Ruled out as explanations: the operator really is `seqbaijkokkos` (`-ksp_view`,
`bs=3`); `MatConvert` appears only in the block run but costs 10 ms against a
116 ms gap; and MatMult call counts (433), hierarchy and converged answer are
identical to the scalar run.

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
