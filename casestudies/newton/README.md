# Case study: Newton on the Bratu problem

`DESIGN.md` was written first and settles three questions; this file reports what
happened when the design was run. `sweep.py` produces the per-variable tables
(`sweep.json`); `search.py` produces the Pareto search (`search.json`).

## Setup

−Δu − λeᵘ = 0 on the unit square, zero Dirichlet data, five-point stencil, Newton
with a direct linear solve (PETSc's SNES ex5 in its simplest form). Three cases
spanning a decade in conditioning; the reference (fp64) converges to a relative
residual of 1e-10 in 3–5 iterations.

| case | k | λ | κ(J) at solution | fp64 iterations |
|---|---|---|---|---|
| 0 | 16 | 1.0 | 123 | 3 |
| 1 | 16 | 6.0 | 265 | 5 |
| 2 | 24 | 6.0 | 575 | 5 |

Variables: the stored Jacobian `J`, the stored step `delta`, the stored residual
`F`, the stored iterate `u`; and, with `fine_residual=True`, the two terms of the
residual `F.lap` and `F.exp` rounded *before* they are subtracted.

**Criterion:** converged to the residual tolerance within an iteration budget
(`IterationBudget`), not relative error against the fp64 run. Convergence is
judged by an independent fp64 residual of the returned iterate, not by the
solver's own stopping test — see "False convergence" below for why that is not
optional.

## Per-variable thresholds

One variable at a time down a virtual-width ladder, everything else fp64.
Predictions are from `DESIGN.md`, made before running.

| variable | predicted mechanism | predicted bits | observed |
|---|---|---|---|
| `J` stored | perturbed operator, η ≈ u_p·κ | knee ≈ log₂κ + 3 ≈ 10–12 | **knee 10±1 for all cases; never diverges** — floor is Picard's count (8 / 39 / 40 vs pure Picard 8 / 39 / 39) |
| `delta` stored | perturbed step, ≤ u_p·κ, typically far less | ≲ log₂κ | +1 iteration at 12 bits, +4–5 at 4 bits; never fails |
| `F` stored (evaluated fp64) | relative RHS perturbation, **κ-free** | very few | 4 bits costs +2–3 iterations, identically across κ |
| `u` stored | caps attainable accuracy, floor ≈ u_p·κ | log₂κ + log₂(1/rtol) ≈ 40–42 | **37–39, and shifts with κ** |
| `F.lap`, `F.exp` before cancellation | cancellation floor | log₂(1/rtol) + 3 ≈ 36 | **33 for all three cases — κ-independent**; below it, false convergence |

Two predictions were confirmed as mechanisms, not just as numbers, because the
design chose cases to discriminate them: `u`'s threshold moves with κ (accuracy
floor ∝ u_p·κ) while the residual terms' threshold does not (cancellation floor
∝ u_p, κ-free). That is the difference between an accuracy-limiting quantity and
a cancellation-limited one, and a single case could not have shown it.

### The Jacobian degrades to Picard

The design predicted divergence once u_p·κ ≳ 1. It never happens. At low
precision the diagonal `4/h² − λeᵘ` rounds to `4/h²` — the nonlinear term is
below the ulp of the Laplacian's diagonal — so the stored Jacobian degenerates to
the Laplacian, and Newton's update `u ← u − L⁻¹(Lu − λeᵘ) = L⁻¹λeᵘ` *is* Picard
iteration. Picard converges for λ below the turning point (λ* ≈ 6.81) at a rate
governed by λ/λ*, which is why the floor is 8 iterations at λ=1 and 39 at λ=6
regardless of grid. Pure Picard reproduces the floors exactly.

So the answer to "how many bits does the Jacobian need" is: **about 10 for this
problem, and below that you pay Picard's rate, not failure.** The knee is where
the smallest physically significant term in a row is resolved against the
largest — log₂ of the row's dynamic range — which is the right diagnostic for a
stiff or multiphysics Jacobian.

### False convergence

With the residual's terms rounded before cancellation, the solver's own stopping
test fires spuriously: near the solution `lap ≈ λeᵘ`, both round to the same
representable value, the computed residual is exactly zero, and the solver
declares victory. In the raw sweep this showed as "converged in 1–2 iterations"
at 4–8 bits. A real solver has only its own residual to look at. This is exactly
why mixed-precision iterative refinement evaluates the residual in high
precision, and it is a rule the tool now enforces: a kernel's `converged` must
come from an independent high-precision residual.

## Pareto search over real storage formats

Slack is the permitted increase in iterations over fp64. Full search, three cases.

**Stored quantities only** (F evaluated in fp64; 4 variables, 256 bits at fp64):

| slack | iters / ref | bits | `F` | `J` | `delta` | `u` |
|---|---|---|---|---|---|---|
| 0    | 1.00 | 147 | tf32 | fp32 | fp32 | fp64 |
| 0.25 | 1.33 | 128 | bf16 | fp16 | fp32 | fp64 |
| 1.0  | 1.67 | 112 | bf16 | fp16 | bf16 | fp64 |
| 4.0  | 1.67 | 112 | bf16 | fp16 | bf16 | fp64 |

**Residual terms rounded before cancellation** (6 variables, 384 bits):

| slack | iters / ref | bits | `F` | `F.exp` | `F.lap` | `J` | `delta` | `u` |
|---|---|---|---|---|---|---|---|---|
| 0    | 1.00 | 275 | tf32 | fp64 | fp64 | fp32 | fp32 | fp64 |
| 0.25 | 1.33 | 256 | bf16 | fp64 | fp64 | fp16 | fp32 | fp64 |
| 1.0  | 1.67 | 240 | bf16 | fp64 | fp64 | fp16 | bf16 | fp64 |

Three things to read off:

- **fp32 Jacobian storage is free** — zero iteration cost. fp16 costs a third
  more iterations. Nothing below fp16 is reachable at any slack: fp8e4m3
  overflows the stencil entries (4/h² ≈ 1156 > 448), and fp8e5m2's two
  significand bits put the solver on Picard's rate, 8× the iterations.
- **`u` is fp64 at every slack**, as an accuracy-limiting quantity must be.
- **Stored `F` goes to bf16 while `F.exp` and `F.lap` stay at fp64, in the same
  row.** The residual may be *stored* in 16 bits; its terms may not be
  *evaluated* in fewer than 33, and the real-format ladder has nothing between
  fp32's 24 and fp64's 53. This is the storage-versus-arithmetic distinction as
  a single table row, and it is the reason a storage-precision tool must never
  be read as answering an arithmetic-precision question.

Prefilter: 72 → 13, 80 → 18, 55 → 14 evaluations (4–5.5×) for 0–5% more bits.

## What this does and does not say about performance

Nothing here is a timing. The design factorizes time as iterations × cost per
iteration; this study measures the first factor. The second is a Kokkos SpMV
kernel timed on hardware for the storage formats above — under one node-hour —
and the factorization is checked at the uniform-fp32 point, where PETSc can be
built with `--with-precision=single` and timed end to end. Until that exists,
"fp32 Jacobian storage is free" is a statement about iterations.
