# Fable brief: what sets the bits a block-inverse line smoother's stored data needs -- a prediction before the measurement

Written 2026-10-10 (Opus 5.5) at Mark's direction ("make a Fable prompt as you suggest").  For a **Fable 5.1**
session started in `~/Codes/parsimony`: *"Read casestudies/kintora/FABLE-BRIEF.md and do it."*

Read `PLAN.md` (this directory) for the study's shape, `DRIVER-CONTRACT.md` for what the hooks will measure, and
`paper/kintora-precision.tex` Sections 3-4 for the solver and the method as the paper states them.  This brief
supersedes PLAN.md's "Fable session" paragraph wherever the two differ.

**Derivation plus a numpy check at L2.  No KINTORA code changes, no A100 time.**

---

## 0. What you are deriving, in one paragraph

KINTORA's Eulerian implicit stage is FGMRES (fp64, rtol 1e-10) on J = M_J^-1 K, right-preconditioned by one V-cycle
of a matrix-free h-multigrid whose smoother is Richardson V(3,3) on a forward + backward line sweep with exact
(cell, v-segment) 24 x 24 block inverses D^-1 and face couplings C to the walk neighbour.  The study will round each
stored quantity -- D^-1, C, the V-cycle's iterate and residual vectors, the coarse operators' tables, the transfer
blocks -- elementwise to k significand bits and measure the outer iterations and the stationary V-cycle's contraction
factor rho.  **Your job is to predict, before any of that runs, the knee k* of each quantity and how it moves with mu,
dt, the v resolution (12 vs 48 segments) and the multigrid level -- and to say what about the operator sets it.**
Stage B (the L2 ladders on the Mac) then tests the prediction rather than being fitted by it.  A prediction that is
wrong in an informative way is a result; a prediction that is not falsifiable is not.

## 1. What is measured already (sources: KINTORA ledger `~/Codes/plasma-kinetics/plans/design-ledger.md`)

Any model must reproduce these.

1. **fp32 D^-1 costs no outer iteration** at L4, L5 and L4 v48 (D230(c)).  Outer counts are 2-4, so this only says
   the degradation is below ~25-50%.  It is one point at k = 24, not a knee.
2. **The 6 x 6 node-block sweep stalls at 48 segments and large mu** (D230(b)): FGMRES flat at 3e-5 on mu row 18 at L4;
   200 its (DIVERGED_ITS) on mu rows 18/23 at L2, dt 0.2, against 2 for the 24 x 24 sweep.  Dropping the intra-segment
   v coupling from the diagonal block is an O(1) perturbation of specific entries of D; rounding D^-1 is an O(2^-k)
   perturbation of every entry.  The question is how close the second comes to the first's effect.
3. **The 24 x 24 sweep is an exact line solve for one-signed streaming** along an arc (D229(b), D230(c): identical outer
   counts to LU line factors at V(2,2), arcs 16).  The coupling *between* adjacent v segments is not in the line solve;
   Richardson resolves it.
4. Defaults (KINTORA 7f8c9d9): 12 v segments x 4 GLL nodes, P2 in the plane (288 unknowns per cell per mu row), 24 mu
   rows, L-series of 16 x 4^L cells, arcs 64 cells on the finest level halved per coarser level, L0 by cuDSS.  ARKIMEX3,
   gamma = 0.4359; the stage matrix K = M_J / (gamma dt) + A_s + mu A_m.  In-plane streaming Courant number 7/16/32 on
   L2/L3/L4 at dt 0.2; mirror Courant <= 0.3.
5. Precedents from Parsimony (paper/parsimony.tex; `casestudies/newton/README.md`; `casestudies/hdcg/NOTES.md`):
   Bratu's stored Jacobian has a knee near 10 bits and degrades to Picard rather than diverging; Grid's coarse Dirac
   stencil rounded to 8 bits left CG at 5 iterations while the attainable residual tracked 2^-k.  A stored residual
   tolerates 16 bits while its terms cannot be evaluated in fewer than 33 (storage is not arithmetic).

## 2. Starting hypotheses -- confirm, refute or sharpen each

These are Opus's guesses, written so they can be wrong.  Do not treat them as the answer.

- **H1, block inverses (componentwise).**  Rounding gives D~^-1 = D^-1 + Delta with |Delta| <= u |D^-1| entrywise,
  u = 2^-k.  The smoother's error propagation S = I - B K changes by dS = -dB K, so ||dS|| <= u || |B| |K| ||: a
  Skeel-type condition number of the blocks, cond_S(D) = || |D^-1| |D| ||, and the sweep's coupling to the walk
  neighbour set the scale.  Predicted knee: k* ~ log2( cond_S * (nu1 + nu2) * A / delta ) for a tolerated relative
  degradation delta of rho, with A the V-cycle's amplification of a smoother perturbation.  Open: is A ~ 1/(1 - rho_S),
  or bounded by the coarse correction?  Where does the first-order model fail -- near the stall regime of fact 2?
- **H2, vectors.**  Rounding the iterate after each smoothing step is an additive perturbation of relative size u per
  step, not a change of rate: a floor on the stationary iteration's attainable residual.  FGMRES is flexible, so a
  preconditioner perturbed at relative size u costs outer iterations only when u approaches the V-cycle's own
  accuracy, ~rho.  Prediction: the vectors need k ~ log2(1/rho) + a few bits, far fewer than D^-1.  (Pointers to check,
  not to cite unread: Saad's flexible GMRES, SISC 1993; Simoncini and Szyld on inexact Krylov methods, SISC 2003.)
  The residual may differ from the iterate: a rounded residual restricted to the coarse grid is a perturbed right-hand
  side of the coarse correction.
- **H3, C with D^-1.**  C enters the same sweep step (r = b - C y, y = D^-1 r), so first-order perturbations add: the
  joint knee is the larger solo knee plus at most about one bit.  This predicts the joint grid (PLAN.md P8).
- **H4, v resolution and mu.**  The 24 x 24 block's v-derivative coupling scales as mu times dL/(dv/2), so it grows with
  mu and with 1/dv: cond_S at 48 segments exceeds cond_S at 12, and the knee shifts by about log2 of their ratio.
- **H5, dt.**  The mass shift 1/(gamma dt) is the block's diagonal ballast: larger dt, weaker diagonal, larger cond_S,
  more bits.
- **H6, levels.**  A coarse level's error is corrected by the finer smoother, so its stored data may tolerate fewer
  bits; but its streaming Courant number is smaller (coarser h), so its blocks are better conditioned anyway.
  Predict which effect dominates, and whether a level's requirement follows its depth below the finest or its mesh.

## 3. What to compute (Mac, numpy/scipy)

All read-only against KINTORA.  Record `git -C ~/Codes/plasma-kinetics rev-parse HEAD` in every output.

a. **Operator at L2 from KINTORA's numpy oracle**, `~/Codes/plasma-kinetics/scripts/eul_proto.py` (import it from
   that path; do not copy it).  `ep.Geom('solovev_L2')`, `ep.EulOp(g, mu, model='bstar', flux='upwind',
   rule='default')`; `op.A = op.As + mu op.Am` (CSR), `op.M` (M_J).  Layout per mu row: [cell in ring-walk order]
   [segment][v node][x node], so a (cell, segment) block is 24 contiguous unknowns.  The oracle agrees with KINTORA's
   residual to 2e-13 (D223(b)).  Its default `VGrid()` is ex12's (12 uniform segments, GLL 4, `-w_nseg 8 -w_degree 2`
   mu rows); 48 segments is `ep.EulOp(g, mu, ..., vgrid=ep.VGrid(nseg=48))`.  The default `upwind='total'` makes the
   trace depend on mu, as ex12's does.  Check one mu row's residual against an ex12 dump (`-eul_dump_row`,
   `scripts/eul_oracle_compare.py`) before trusting a number.
b. **Block conditioning**: for mu rows 0, 12, 18, 23 and dt 0.05, 0.2, 0.4, at 12 and 48 segments: the distribution
   (min, median, max) of cond_2 and cond_S of the 24 x 24 diagonal blocks of K, and where in the plane and in v the
   worst blocks sit.
c. **The smoother alone, with rounding**: build KINTORA's sweep as the code does it (`EulLinesSweep24` and
   `KinEulOpDevPBSetUp` in `src/kin/eul/kokkos/kin_eulop.kokkos.cxx`; the arcs from `KinEulOpDevLinesMFSetUp`; the
   fused Richardson in `EulMFLinesRichardson`, `src/kin/eul/kin_eulmg.cxx`) as a scipy LinearOperator; measure the
   contraction of Richardson(3) on it (power iteration on the error propagation, or a residual history on random
   initial errors) with D^-1 and C at fp64 and rounded to v24, v16, v11, v8 (use `parsimony.formats.round_to`).  This
   is the smoother-only ladder: it tests H1, H3-H5 on the smoother before the V-cycle hooks exist.  The full V-cycle is
   not in the oracle; say what the smoother-only ladder can and cannot predict about it.
d. **One calibration point from the real V-cycle**: ex12 on the Mac at L2, `-eul_mg_solve -eul_mg_device -eul_mg_mf
   -eul_mg_cycles 20`, `-eul_mg_sweep_fp32 1` against `0`, base set (`-eul_mg_rows 0,12,23 -eul_mg_dts 0.05,0.2,0.4`) and
   stiff set (`-v_nseg 48 -eul_mg_rows 18,23 -eul_mg_dts 0.2,0.4`).  The base options are `scripts/check_ex12_eul.sh`'s
   `BASE` with `-pol_mesh_file .../solovev_L2.mesh`.  **Copy `src/sldg/ex12` to your scratchpad and run the copy** with
   absolute mesh paths; never build in that tree (another session works in it).  This gives the measured
   delta-rho at k = 24 per case, the one real number the model is calibrated to.

## 4. Deliverables

1. `casestudies/kintora/SMOOTHER-BITS.md`: the derivation (assumptions stated, the model per quantity, where it fails
   and why), the L2 results of 3b-3d, and **the prediction table**: per quantity and case, the predicted knee k* (at
   delta = 10% of rho, and the k at which an outer iteration is first added), the predicted shift from 12 to 48
   segments, and what Stage B result would falsify each line.  Write it so it can be read without this brief.
2. `casestudies/kintora/bits_model.py`: the computation of section 3, runnable, writing `results/bits_model-L2.json`
   (KINTORA commit, options, every number in the note).
3. In `paper/kintora-precision.tex`: replace the Section 5 `\pending` boxes' "Prediction to test" lines with the
   predictions and their basis, and draft Section 4.5's item 2 (the blame-style analysis) to match the model you
   derived.  Prose per `~/Codes/plasma-kinetics/paper/house-style.md`: "mu row", never a bare "row"; every number
   with its source; a conjecture labelled as one.
4. If the model yields a predictor worth having in the library (for instance a componentwise condition number as the
   "blame" coefficient of PLAN.md P7), a short spec for it in PLAN.md's library track -- the spec, not the code.
5. New findings for KINTORA (for instance, blocks near singular at some (mu, dt), or a smoother margin that motivates
   the v-coupling sweep of session 8 item 7) as entries in `FINDINGS-FOR-KINTORA.md`, status "drafted".

## 5. Rules

- **Do not edit `~/Codes/plasma-kinetics`** (code, ledger, memory): findings go through `FINDINGS-FOR-KINTORA.md` and
  Mark.  Read, import, and run copies only.
- Fable is a limited resource (Mark's rule): if the derivation shows the answer is simple -- one quantity, a knee set
  by log2(cond) with no dependence worth a model -- say so in the note and stop.  Do not spend the session on work
  Opus can do (plotting, table formatting, the library code).
- Verify positively: a run that wrote no output is a broken run, not a pass; check the marker line
  (`eulmg solve ... V-cycle factor`) is present before reading a number.
- Parsimony tests stay green (`.venv/bin/pytest`).  Commit locally with the trailer of the model that did the work
  (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`); push only when Mark says.
- Before context runs low: write the note's current state and the exact next step, then a handoff in `HANDOFF.md`.

## 6. Read first, in order

1. `casestudies/kintora/PLAN.md`, then `DRIVER-CONTRACT.md`.
2. `paper/kintora-precision.tex`, Sections 3 and 4.
3. KINTORA ledger D229(b), D230(b), D230(c) (grep the IDs in `~/Codes/plasma-kinetics/plans/design-ledger.md`; the
   file is 0.7 MB, do not read it whole).
4. `~/Codes/plasma-kinetics/src/kin/eul/kokkos/kin_eulop.kokkos.cxx`: `KinEulOpDevPBSetUp` (from line ~1666),
   `EulLinesSweep24` and `KinEulOpDevLinesSweep` (~2450-2580); `src/kin/eul/kin_eulmg.cxx`: `EulMFLinesRichardson`,
   `EulMFResidual`, `KinEulMGSetUpKSP`; `src/kin/gates/kin_gate_eul.cxx`: `EulMGSolveGate` (how `-eul_mg_cycles`
   computes rho).
5. `~/Codes/plasma-kinetics/scripts/eul_proto.py` (header and `checks()`).
6. `src/parsimony/formats.py` (`round_to`), `src/parsimony/criteria.py`, and `casestudies/newton/DESIGN.md` section 1
   (rate-limiting against accuracy-limiting quantities).
