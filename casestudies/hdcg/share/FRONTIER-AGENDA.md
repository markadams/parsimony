# Grid precision study on Frontier

Mark Adams, LBNL · mfadams@lbl.gov · 2026-10-05

Goal: measure how few bits HDCG's coarse grid can be stored in, on a real configuration on Frontier, starting from Peter's `Example_hdcg_multigrid` on `develop` (`87baaf1d`).

## Test problems, in order

1. **Build check:** `Example_hdcg_multigrid` with `NBASIS=8`, hot configuration, 1 node.
2. **HDCG on a real configuration, at a size we can sweep.** 48^3x96 needs 36 nodes per point; something like 24^3x64 on 1-4 nodes, with a matching subspace file.
3. **Baseline:** fp64 coarse space against `COARSE_SINGLE` (fp32), to reproduce Peter's result first.
4. **16-bit coarse stencil storage:** bf16, and fp16 with a per-block scale.
5. **Peter's production `pvdagm_multigrid.job` once, unmodified,** to confirm we can run it and to measure the dense coarsest level's memory and time.
6. **Joe's PETSc coarse space on the same configuration and operator,** so HDCG and AMG are compared on one problem.

## Coarse-grid storage

**HDCG coarse stencil:** block-sparse, not dense. It has 81 points (offsets in {-1,0,1}^4), each a dense 62x62 complex block. Grouped by how many coordinates of the offset are nonzero, the 81 points split 1 / 8 / 24 / 32 / 16.

- Measure each block's norm by group first. The 48 corner blocks (3 or 4 nonzero coordinates) couple only sites near block corners, so we expect them to be small.
- If they are small, drop whole blocks (81 to 33 points). That keeps the GEMM structure, needs no index arrays, and sends fewer halo messages.
- Do not drop entries inside blocks. Each kept entry would need a 4-byte index beside an 8-byte value, so more than 1/3 of entries must go just to break even. Lowering precision is cheaper.
- Precision and truncation can be searched jointly, with a different precision for near and far blocks.

**PVdagM coarsest level (`DenseCoarseMatrix`):** genuinely dense. It stores an explicit inverse, built in fp64 and rounded once to an fp32 slab.

- This is the best target for lower precision. A dense GEMM moves no indices, so storage below fp64 gains exactly 8/w: 4x at 16 bits, 8x at 8 bits. The coarse solve tolerance is 0.05.
- Later: low-rank compression of the inverse (block low-rank or HSS, as in STRUMPACK).

## What the test problem needs

- A coarse storage precision switch in the XML (for example `<CoarseStorage>bf16</CoarseStorage>`), so the study runs Peter's binary, not a patched copy.
- Machine-readable output: iterations, true residual, time per phase, ranks and `--mpi` layout.
- Saved subspace and eigenvector files per configuration, so each point skips setup.
- 3 decorrelated configurations from one ensemble.
- `FineSloppyComms` on and off, as a variable for reduced-precision communication.

## Frontier setup this week

- [ ] OLCF accounts on the project for Mark and Joe
- [ ] Read access to Peter's configurations and subspace files under `phy157`, or a copy in our project
- [ ] Build from `systems/Frontier/sourceme-rocm7.2.sh` and `config-command`: c-lime (spack), gmp, mpfr, openssl. Reuse Peter's spack environment if possible.
- [ ] Build check (test problem 1) on 1 node

## Questions for Peter and Joe

1. Is there a smaller configuration and subspace file on Frontier that fits 1-4 nodes?
2. Have you tried HDCG with the `Nearest` or `NextToNearest` coarse geometry instead of the 81-point box?
3. Would a coarse storage precision switch in the XML be welcome upstream?
4. Do real-configuration coarse stencil entries fit fp16's exponent range, or is bf16 the only sensible 16-bit format?
5. How tolerant is the fp32 dense slab of further rounding, given the coarse solve tolerance of 0.05?
6. Which project and directory should the three of us share on Frontier?
