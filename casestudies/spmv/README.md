# Case study: SpMV cost per storage format

Tier 2 of the cost model in [`../newton/DESIGN.md`](../newton/DESIGN.md). The
search tiers count iterations; this one measures what an iteration costs as a
function of storage format, so that

    time  =  iterations (tier 1)  ×  cost per iteration (tier 2)

has a measured second factor. Nothing here has been run on hardware yet; the byte
model and the matrix are checked on the host.

```
c++ -std=c++17 -O2 test_stencil.cpp -o test_stencil && ./test_stencil   # no GPU
PETSC_ARCH=<a kokkos-cuda arch> ./build.sh
sbatch run_perlmutter.sh
```

The SpMV sweep needs only a Kokkos+CUDA arch — it templates its own storage
types, so a single-precision PETSc build is not required. That build is for the
separate end-to-end check of the factorization at the uniform-fp32 point.

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

Blocking lifts the cap. For a matrix shaped like PETSc's
`snes/tutorials/ex56` — 3D linear elasticity on trilinear hexahedra, `bs = 3`,
27 blocks per block row — the model gives:

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
