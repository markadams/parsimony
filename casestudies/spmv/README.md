# Case study: SpMV cost per storage format

Tier 2 of the cost model in [`../newton/DESIGN.md`](../newton/DESIGN.md). The
search tiers count iterations; this one measures what an iteration costs as a
function of storage format, so that

    time  =  iterations (tier 1)  ×  cost per iteration (tier 2)

has a measured second factor. **Measured on an A100-SXM4-80GB**, 2026-09-22;
raw output in [`results-a100.txt`](results-a100.txt), job took 36 seconds.

```
c++ -std=c++17 -O2 test_stencil.cpp -o test_stencil && ./test_stencil   # no GPU
PETSC_ARCH=<a kokkos-cuda arch> ./build.sh
sbatch run_perlmutter.sh
```

The SpMV sweep needs only a Kokkos+CUDA arch — it templates its own storage
types, so a single-precision PETSc build is not required. That build is for the
separate end-to-end check of the factorization at the uniform-fp32 point.

## Measured results

Same elasticity matrix (`bs=3`, 27 blocks/block-row, 512k block rows, 121M scalar
nonzeros), stored two ways, on one A100. Peak HBM2e is 2039 GB/s. "Realized" is
the fraction of the byte model's predicted gain that actually appeared.

| layout | format | ms | GB/s | % peak | measured | model | realized |
|---|---|---|---|---|---|---|---|
| AIJ (bs=1)  | fp64 | 1.094 | 1358 | 67% | — | — | — |
| AIJ  | fp32 | 0.902 | 1110 | 54% | 1.21× | 1.48× | 44% |
| AIJ  | fp16 | 0.811 |  936 | 46% | 1.35× | 1.96× | 36% |
| **BAIJ (bs=3)** | fp64 | 0.668 | 1573 | **77%** | — | — | — |
| **BAIJ** | fp32 | **0.382** | 1482 | 73% | **1.75×** | 1.86× | **87%** |
| **BAIJ** | fp16 | 0.290 | 1115 | 55% | 2.30× | 3.25× | 58% |

Against the scalar fp64 baseline — the thing a PETSc user actually has today:

| | measured | model |
|---|---|---|
| BAIJ fp64 (blocking alone) | **1.64×** | 1.41× |
| BAIJ fp32 | **2.86×** | 2.63× |
| BAIJ fp16 | 3.77× | 4.58× |

### What it says

**Blocked fp32 is the real win, and it beats the model.** BAIJ fp32 realizes 87%
of its predicted gain, and against scalar fp64 it delivers 2.86× where the byte
model predicted 2.63×. Blocking *over*-delivers because it does not only move
fewer bytes — it moves them better, reaching 77% of peak bandwidth against AIJ's
67%. Fewer, larger, more regular loads.

**The 2013 result survives for scalar formats.** On the 7-point stencil, fp32
buys 1.07× — 19% of its predicted 1.37×, which is to say nothing. That is the
same "don't bother" Blame Analysis reported, on 2026 hardware. At 7 nonzeros per
row the kernel is latency-bound, not bandwidth-bound, and narrowing the values
does not help. Scalar 27-point does better (1.27×) but still realizes only 59%.

This sharpens the paper's thesis. It is **not** simply that hardware changed. On
the scalar formats the original tools targeted, reduced precision still buys
almost nothing on a thin stencil. What unlocks the payoff is the *blocked
format* — a data-structure choice, not a hardware one.

**fp16 underdelivers everywhere**, realizing 58% even in BAIJ. Achieved bandwidth
collapses from 1573 to 1115 GB/s: the kernel is moving less data but not
proportionally faster, so below about 4-byte values it has left the
bandwidth-bound regime and per-element instruction cost dominates. Part of that
may be this kernel rather than the hardware — we do not claim the fp16 figure is
the best achievable — but the direction is clear, and it means the format-width
ratio is an upper bound that fp32 approaches and fp16 does not.

### Storage, not tensor cores

fp16 and bf16 here are **storage formats only**. The kernel loads 2-byte values
and converts to `double` for the multiply-add, so the arithmetic is scalar FP64
throughout. Disassembling the binary confirms it: zero `HMMA`/`IMMA`/`BMMA`
instructions, 50 `DFMA`, no `FFMA`, and 44 `F2F` conversions.

This is not an oversight. Tensor cores operate on dense tiles — 16×16×16 at
smallest for the half-precision instruction — so a 3×3 elasticity block fills 9
of 256 slots, and NVIDIA's tensor-core sparse path requires block sizes of 16 or
more for the same reason. At `bs = 3` the matrix units are simply unreachable.
The payoff measured here is memory traffic, and nothing else.

The 44 conversion instructions are also the likely reason fp16 underdelivers:
once the value traffic is small enough, per-element conversion and issue cost
dominate. Accumulating in fp32 rather than fp64 after the load would halve the
conversion width and use `FFMA` (19.5 vs 9.7 TFLOP/s on A100); whether that
closes the gap is an open question and a cheap experiment.

### Correctness

BAIJ and its scalar expansion return **identical checksums at every format** —
1.88300e+06 (fp64 and fp32), 1.87715e+06 (bf16), 1.88282e+06 (fp16). The two
layouts hold the same matrix and the kernel computes the same thing from both,
which is what makes the layout comparison a comparison and not two experiments.
bf16 and fp16 differ from fp64 in the 4th and 5th digit respectively, as their
significands predict.

## The model, and why blocking is the interesting axis

Per block row, block CSR with block size `b` moves

    r_b · (b²·w + 4)  +  4  +  b · (v + 8)    bytes

for `r_b` blocks per block row, value width `w`, vector width `v`. Narrowing `w`
leaves three terms alone: the column indices, the row pointer, and the vector
read and write. A block carries **one index for b² values**, so the index costs
`4/b²` bytes per scalar nonzero and the saving is bounded by

    (8 + 4/b²) / (w + 4/b²)

| block size | index bytes / scalar nnz | fp32 bound | fp16 bound |
|---|---|---|---|
| 1 (AIJ) | 4.00 | 1.50× | 2.00× |
| 2 | 1.00 | 1.80× | 3.00× |
| 3 (elasticity) | 0.44 | 1.90× | 3.45× |
| 6 | 0.11 | 1.97× | 3.84× |

So in scalar CSR, narrowing values **cannot** halve the time — the 4-byte column
index per nonzero caps it at 1.5×, and thin rows do worse still (1.36× at a
7-point stencil). This is worth stating plainly because the format width invites
the opposite assumption.

### The ex56 stencil, measured rather than assumed

The block size and stencil were taken from real runs of `snes/tutorials/ex56`
(3D linear elasticity, `-run_type 1`, `-ksp_view_mat ::ascii_info`), not guessed.
PETSc reports `bs=3`, and on a d³ grid of interior nodes the block count follows
a closed form that reproduces every measured run exactly:

| element | blocks | d | measured blocks/row | closed form |
|---|---|---|---|---|
| Q1 | (3d−2)³ | 7 | 20.0 | 20.0 |
| Q1 | | 11 | 22.4 | 22.4 |
| Q2 | (4d−5)³ | 15 | 49.3 | 49.3 |
| Q2 | | 23 | 54.1 | 54.1 |

So blocks per block row → **27 for Q1** and **64 for Q2** as the mesh refines,
and scalar nnz/row → 81 and 192.

Usefully, the bandwidth conclusion barely depends on which of these you pick:

| stencil | AIJ fp32 | BAIJ fp32 | AIJ fp16 | BAIJ fp16 | BAIJ fp32 vs AIJ fp64 |
|---|---|---|---|---|---|
| Q1, d=11 (22.4) | 1.48× | 1.85× | 1.95× | 3.22× | 2.62× |
| Q1 asymptotic (27) | 1.49× | 1.86× | 1.96× | 3.26× | 2.63× |
| Q2, d=23 (54.1) | 1.49× | 1.88× | 1.98× | 3.35× | 2.66× |
| Q2 asymptotic (64) | 1.49× | 1.88× | 1.98× | 3.37× | 2.67× |

Once `bs = 3` the index term is already small, so the result is robust to getting
the stencil wrong — which is worth knowing, since the stencil is the parameter a
reader is most likely to dispute.

### The table, at Q1

For a matrix shaped like `ex56` at Q1 — `bs = 3`, 27 blocks per block row — the
model gives:

| | AIJ bytes/block-row | BAIJ bytes/block-row | AIJ gain | BAIJ gain |
|---|---|---|---|---|
| fp64 | 2976 | 2104 | 1.00× | 1.00× |
| fp32 | 2004 | 1132 | 1.48× | **1.85×** |
| fp16 | 1518 | 589 | 1.96× | **3.24×** |

- Blocking alone, at fp64: **1.41×** less traffic than AIJ.
- BAIJ at fp32 against AIJ at fp64: **2.62×** less traffic.
- Narrowing only the *vector*: 1.06×, since its traffic is O(n) not O(nnz).

Two consequences. Reduced precision and blocking are complementary, not
alternatives. And a tuning tool that reports "fp32 suffices" has not predicted a
speedup — the payoff depends on a storage decision the tool never sees, which is
why the design keeps precision search and cost measurement apart.

## What is measured

`spmv.cpp` times a block CSR SpMV templated on `(MatT, VecT, BS)`, accumulating
and writing in fp64. `BS = 1` is scalar CSR, so AIJ and BAIJ share one code path.

The elasticity matrix is timed **both ways** — blocked, and expanded to scalar by
`expand_to_scalar()` — so the AIJ/BAIJ difference is layout alone rather than two
different problems. `test_stencil.cpp` checks the expansion entry by entry.

Each configuration is swept over team sizes {4, 8, 16} and vector lengths
{4, 8, 16, 32} and the best taken, since Kokkos' `AUTO` misbehaves for this
access pattern. A checksum is reduced from the result so a format that silently
produced garbage is visible, and so the kernel cannot be optimized away.

## Host checks

`test_stencil.cpp` runs without Kokkos or a GPU and asserts, before any
allocation is spent:

- Row structure, column ranges, no duplicate columns, **sorted columns** (PETSc
  requires them, and they give the input vector its natural access order).
- Symmetry and strict diagonal dominance of the assembled matrix, including the
  full `bs × bs` diagonal block.
- **No entry exactly representable in 4 significand bits.** A constant-coefficient
  Laplacian has entries 4 and −1, exact in every format down to 8 bits, which
  makes matrix storage look free — the trap documented in
  [`../nas_cg`](../nas_cg).
- The byte model against its derivation, and against the fat-row bound for each
  block size.
