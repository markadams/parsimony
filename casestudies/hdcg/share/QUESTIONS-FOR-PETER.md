# Grid questions — precision study on the coarse space

Mark Adams, LBNL · mfadams@lbl.gov · 2026-09-28
Against Grid `develop 865c338d`, Perlmutter, CUDA / sm_80, 4× A100.

Numbered so you can reply inline. **Q1 is the one that blocks us.**

---

## Blocking

**Q1. Which driver should a coarse-space precision study start from?**

`tests/debug/Test_general_coarse_hdcg.cc` needs
`MultiGeneralCoarsenedMatrix(geom, CoarseMrhs)` followed by `CoarsenOperator`.
That class is now in `Grid/algorithms/multigrid/deprecated/`, and the block
that used it in `Test_general_coarse.cc` is inside `#if 0`.

It segfaults inside the constructor, immediately after
`NonLocalStencilGeometry has 81 terms in stencil`, under every combination we
tried:

| varied | values tried | result |
|---|---|---|
| ranks | 1, 2, 4, 8 (1 and 2 nodes) | all segfault |
| gauge | unit, hot | all segfault |
| subspace | `CreateSubspaceChebyshev` (ord 500 and 20), `CreateSubspaceRandom` | all segfault |
| nbasis | 40, 20 | all segfault |
| `--device-mem` | none, 2000, 8000, 16000, 32000 | all segfault |
| comms flags | with/without `--shm 2048 --comms-sequential --shm-mpi 0` | all segfault |

Rounding instrumentation is off in every one of those runs, so it is not ours.
Is that path expected to work on develop? If it is retired, should this move to
`MultiGeneralCoarsenedOperatorV2` / `HDCGMultiGrid.h` — and is there a driver to
copy, or is writing one the first job?

**Q2. What do you run `Test_general_coarse_hdcg` on?**

The stock configuration (Ls 16, nbasis 40, 4⁴ blocks, 16³×32) does not fit four
A100s. Grid's own `Test_general_coarse.cc` at 1 rank dies with

```
cudaMalloc failed for 8153726976 out of memory
```

and passes at 4 ranks. Node count, and anything beyond
`systems/Perlmutter/config-command`, would save us guessing.

**Q3. Where is `ckpoint_lat.4000` on Perlmutter?**

A path is enough. Everything below is free-field because we do not have it.

---

## About the measurement

We rounded the *stored* coarse Dirac stencil to `k` significand bits
(round-to-nearest, float64 exponent range), kept all arithmetic in fp64, solved
with CG against the rounded operator, and judged the answer against an exact
second coarsening of the same aggregation.

Preliminary: free field (unit gauge), random subspace, nbasis 20, Ls 8, 16³×32,
CG tol 1e-8, 4× A100. Two RHS cases agree to three digits.

| stored as | sig. bits | δ stencil | true resid | CG iters |
|---|---|---|---|---|
| fp64 | 53 | 0 | 3.089e-09 | 5 |
| v40 | 40 | 3.88e-13 | 3.089e-09 | 5 |
| v30 | 30 | 3.97e-10 | 3.100e-09 | 5 |
| v24 / **fp32** | 24 | 2.54e-08 | 1.834e-08 | 5 |
| v20 | 20 | 4.07e-07 | 2.920e-07 | 5 |
| v16 | 16 | 6.51e-06 | 4.654e-06 | 5 |
| v12 | 12 | 1.04e-04 | 7.476e-05 | 5 |
| v10 | 10 | 4.17e-04 | 3.024e-04 | 5 |
| v8 / **bf16** | 8 | 1.78e-03 | 1.195e-03 | 5 |
| **fp16** | 11, 5 exp | **1.000** | 1.500e-04 | 5 |

δ is the relative L2 change the rounding caused, logged as a positive check that
the hook fired. `fp32` reproduces `v24` and `bf16` reproduces `v8` to every
digit, which is our confirmation that the emulated widths mean what the named
formats mean.

**Q4. Does the flat iteration count match what you would expect of a physics coarse space?**

Five iterations at every format, down to eight significand bits, while the
attainable residual tracks 2^-k exactly. We read that as: the coarse operator is
rate-limiting, not accuracy-limiting, so inside a preconditioner its stored width
should be nearly free. The free field and a random subspace make the conditioning
benign — does this survive a real ensemble?

**Q5. Is 16-bit *storage* of the coarse stencil worth doing?**

Storage only, fp64 arithmetic — orthogonal to the 16F/BF16 *compute* types in
`GridBLAS`. The apply is bandwidth-bound at production batch (AI ≈ 8.7 flop/B at
nbasis 62, nrhs 12), so halving the stencil stream again looks like the next
lever after batching. Interesting to Grid, or is fp32-vs-fp64 the only
distinction that matters in practice?

**Q6. If so, bf16 rather than fp16?**

fp16 destroyed the stencil outright — δ = 1.000, every entry underflowed the
5-bit exponent and the operator rounded to zero. bf16, with *three fewer*
significand bits but fp32's exponent range, was indistinguishable from the 8-bit
virtual format. Our entries are unusually small because of the free field, so the
size of that effect is partly an artifact of the test — but do production coarse
stencil entries sit comfortably inside fp16's exponent range, or is bf16 the only
sane 16-bit choice?

**Q7. Which HDCG inputs would you predict to be most and least tolerant?**

The plan measures three: the near-null subspace, the coarse stencil, and the
deflation eigenvectors. A prediction recorded before the measurement is worth
more than one after it.

---

## PETSc side

**Q8. Which Grid commit should PETSc pin, and with what configure flags?**

`config/BuildSystem/config/packages/grid.py` pins `da593796` (2024-03): no
`multigrid/` directory, and the only flag passed is `--enable-comms=mpi-auto`,
so it is a CPU-only GEN SIMD build. For a performance build through PETSc, what
should the SIMD target, accelerator and shm settings be?

**Q9. Is a portable reference shell worth having alongside a Grid-backed one?**

On the PETSc side the lattice operators could be a parallel-correct,
Hermitian-transpose-aware DMPlex shell as the portable reference, with a second
shell calling Grid directly in Grid's own layout. Useful, or duplicated effort?

---

## Two patches, if wanted upstream

**Q10. `Grid/perfmon/PerfCount.h` — build break on current NERSC defaults.**

Under `GRID_CUDA` Grid defines no-op `__rdtsc` / `__rdpmc` shims to keep
`x86intrin.h` away from nvcc, but cudatoolkit 13.2's header chain declares the
real `__rdtsc` first:

```
PerfCount.h(54): error: cannot overload functions distinguished by return type alone
```

(`unsigned long long` against `uint64_t`). Both uses are in `Stat.cc`, host-side,
never in a device kernel, so replacing the shims with `#include <x86intrin.h>`
builds and restores real timings. Note cudatoolkit 12.4 is not an escape: it
refuses gcc-native/14 outright, which is the Perlmutter default.

**Q11. `AdefMrhs.h` — `TwoLevelCGmrhs` prints its iteration count but does not expose it.**

Added `IterationsToComplete`, `TrueResidual` and `ConvergedFlag`, filled on both
the converged and non-converged exits, exactly as `ConjugateGradient` already
does. Without it, any automated study has to scrape stdout.

---

Raw JSON records, driver sources and the build recipe are in
`parsimony/casestudies/hdcg/`. Slides and an interactive walkthrough of the
pipeline are attached as self-contained HTML.
