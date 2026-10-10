# Findings for KINTORA

The log of what the Parsimony study gives back to KINTORA (`PLAN.md`, "How the two projects inform each other").
One entry per finding: evidence, the proposed KINTORA action, status.  Status values: drafted, sent (to the active
KINTORA session, via Mark), adopted / rejected (with the KINTORA ledger entry or commit that records it).
Parsimony never edits KINTORA's tree or ledger; a KINTORA session does.

## K1 (2026-10-10) -- the evidence for fp32 inverses is the outer count only

- **Evidence.** D230(c) supports "fp32 inverses cost nothing" with identical outer FGMRES counts at L4, L5, L4 v48.  At
  2-4 its a count changes only when the preconditioner degrades by 25-50%.
- **Action.** `-eul_mg_cycles 20` contraction factor, `-eul_mg_sweep_fp32 1` vs `0`, at L4 v48 mu rows 18/23,
  dt .2/.4, and at L6; record the factors in the ledger either way.
- **Status.** Sent 2026-10-10 (Mark relayed it to KINTORA session 8).

## K2 (2026-10-10) -- `d_pbinv` is resident set-up scratch on the default path

- **Evidence.** With kernel 3 and `-eul_mg_sweep_block 24`, `KinEulOpDevLinesSweep` returns through
  `EulLinesSweep24`, which reads `d_sw24`/`d_sw24f`, `d_sweepg` and `d_mlmeta` only.  The readers of `d_pbinv` are
  `KinEulOpDevPBSetUp` itself (the uninverted node blocks the 24 x 24 team kernel assembles from),
  `KinEulOpDevPBApply`, the GMRES line kernels 1/2, the 6 x 6 sweep, and gate E0.  The `PetscCheck` at the top of
  `KinEulOpDevLinesSweep` requires it to exist.  Size: 6 values per unknown, fp64: 0.30 GB at L5 (L1..L5),
  1.21 GB at L6.
- **Action.** Assemble the node blocks in team scratch in the PBSetUp24 rewrite (session 8 item 2), or free the
  array after set-up on this path.
- **Status.** Sent 2026-10-10.

## K3 (2026-10-10) -- `d_Minv` on every level, read on the finest only

- **Evidence.** `KinEulOpDevSetUp` allocates it per level; `KinEulOpDevMassPtr` (its reader) is called on the finest
  level's operator (`EulMGJMult`, `kin_eulstep.cxx`).  The coarse copies: 6 values per unknown, ~0.30 GB at L6.
- **Action.** Allocate on the finest level only, after checking no coarse-level path calls the inverse mass.
- **Status.** Sent 2026-10-10 (minor).

## K4 (2026-10-10) -- after fp32 inverses, the couplings C are the largest fp64 array

- **Evidence.** `d_sweepg` stores 12 values per unknown at fp64 = 0.60 GB at L5 (L1..L5), 2.42 GB at L6, equal to
  the fp32 inverses.  If the sweep becomes bandwidth-bound (session 8 item 1), C is half its stored bytes.
- **Action.** Load C through a template parameter as the sweep's `DT` loads D^-1, so its storage type can change
  without a rewrite.  No format chosen yet; the study measures it.
- **Status.** Sent 2026-10-10.

## K5 (2026-10-10) -- the D^-1 hook site is the PBSetUp24 store

- **Evidence.** `DRIVER-CONTRACT.md`: the `dinv` hook is the store of the inverse in the `KinEulOpDevPBSetUp24`
  kernel, which session 8 item 2 rewrites.
- **Action.** Keep that store a single assignment-from-double per entry; if the rewrite is pivot-free, record whether
  the 24 x 24 blocks are diagonally dominant (the study wants their condition numbers).
- **Status.** Sent 2026-10-10.

## K6 (2026-10-10) -- the Eulerian paper's Sec. 2 describes the superseded design

- **Evidence.** `paper/kintora-eulerian/kintora-eulerian.tex` Sec. 2: defect correction on `L_1` (BAIJ 4 x 4, ring
  LU, 6 segments, a 130 GB TODO); D222(c) found it cannot converge, and the code runs the multigrid on the P2 x P3
  operator with the 24 x 24 sweep.
- **Action.** Rewrite Sec. 2 when that paper is next touched.
- **Status.** Sent 2026-10-10 (low priority).
