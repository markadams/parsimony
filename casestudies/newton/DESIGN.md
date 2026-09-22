# Design: precision of a Newton–Krylov solver

This settles three questions that were open when the conjugate-gradient study
finished, and records the reasoning so the experiments that follow do not have
to rediscover it. Written before the experiments, then checked against them
(see "Predictions" and `results.json`).

## 1. The observable is iteration count, not error

The naive question — "how many bits does the Jacobian need?" — is not well
posed, and the CG study already showed why (its finding 4): near a solver's own
convergence level, relative error against a full-precision run measures
bit-reproduction of a rounding trajectory, not solution quality.

For Newton the situation is sharper still. Write the step with a stored,
rounded Jacobian as $s = -(J+E)^{-1}F$. Then

$$F + Js = E\,(J+E)^{-1}F, \qquad \eta := \frac{\|F + Js\|}{\|F\|} \;\lesssim\; \|E\|\,\|J^{-1}\| \;\le\; u_p\,\kappa(J),$$

where $u_p = 2^{-p}$ for $p$ significand bits and $|E_{ij}| \le u_p |J_{ij}|$.
This is exactly the inexact-Newton forcing term of Dembo–Eisenstat–Steihaug.
Two consequences follow from the standard theory (Kelley, Thm. 5.4.1 and its
corollaries):

* **The converged solution does not depend on $J$'s precision at all**, so long
  as the residual $F$ is evaluated accurately and the iteration converges. $J$
  only chooses the direction; $F$ is what is driven to zero.
* **What changes is the rate.** Quadratic convergence becomes q-linear with
  contraction factor $\approx \eta \approx u_p\,\kappa(J)$. The iteration stalls
  or diverges once $u_p\,\kappa(J) \gtrsim 1$.

So an error-based acceptance criterion would report "the Jacobian needs zero
bits", which is true and useless. The right observable is **cost at fixed
accuracy**: hold the residual tolerance fixed (that is what a solver does
anyway), and measure iterations. This inverts Precimonious' framing — accuracy
is fixed by the stopping criterion, and precision trades against iterations —
and it is the framing the CG study's finding 4 was asking for.

The same argument covers every quantity in an iterative method that is not the
residual or the solution. A preconditioner in low precision is a different
fixed linear operator: it changes the iteration count and cannot change the
fixed point. Inner Krylov vectors likewise. The design principle:

> For iterative methods, quantities divide into *accuracy-limiting* (the
> residual as evaluated, the solution as stored) and *rate-limiting* (the
> Jacobian, the preconditioner, the search directions, the step). One criterion
> catches both: **converged to the residual tolerance within an iteration
> budget**. Accuracy-limiting failures appear as non-convergence; rate-limiting
> costs appear as extra iterations.

### Predictions (to be checked, not assumed)

For a variable $v$ stored at $p$ bits, with everything else at full precision:

| variable | mechanism | predicted contraction | bits needed |
|---|---|---|---|
| $J$ (stored) | perturbed operator | $u_p\,\kappa(J)$ | $\approx \log_2 \kappa(J) + $ margin |
| step $\delta$ (stored) | perturbed update | $\le u_p\,\kappa(J)$, typically far less | $\lesssim \log_2 \kappa$ |
| $F$ (stored after fp64 evaluation) | relative perturbation of RHS | $u_p$, **no $\kappa$** | very few |
| $F$ (terms rounded before cancellation) | cancellation in $-\Delta u - \lambda e^u$ | floor $\approx u_p\,\kappa$ on attainable residual | $\log_2\kappa + \log_2(1/\mathrm{rtol})$ |
| $u$ (stored) | caps attainable accuracy | floor $\approx u_p\,\kappa$ | $\log_2\kappa + \log_2(1/\mathrm{rtol})$ |

Concretely: iterations should stay at the fp64 count while
$p \gtrsim \log_2 \kappa(J) + 3$, grow like $\log(1/\mathrm{rtol}) / \log(1/(u_p\kappa))$
below that, and fail when $u_p \kappa \gtrsim 1$. For a Bratu Jacobian with
$\kappa \sim 10^2$–$10^3$ the knee is at 8–12 bits: **fp16 storage of $J$ should
be free, bf16 marginal, fp8 fatal.** If the measured knee does not track
$\log_2 \kappa$ across cases, the model is wrong and the study stops until we
understand why.

### Outcome (checked 2026-09-22; details in README.md)

Four of five predictions held as mechanisms; one was wrong in an instructive way.

* $J$: knee at 10±1 bits for all cases — within the predicted band. **But it
  never diverges.** The worst-case bound $u_p\kappa \gtrsim 1$ is not reached
  because the rounding error is not aligned with $J$'s small singular vectors;
  instead the nonlinear diagonal term falls below the ulp of the Laplacian's
  diagonal and the stored Jacobian degenerates to the Laplacian. Newton with
  $J := L$ is Picard iteration, which converges below the turning point at a
  rate set by $\lambda/\lambda^*$. Pure Picard reproduces the low-precision
  floors exactly (8 / 39 / 39 against 8 / 39 / 40). The prediction "fp8 fatal"
  was therefore wrong: fp8 is *slow*, at Picard's rate.
* $\delta$: gradual, never fails, no $\kappa$ signature. As predicted.
* Stored $F$: $\kappa$-free, 4 bits costs +2–3 iterations. As predicted, exactly.
* $u$: fails below 37–39 bits and the threshold shifts with $\kappa$. Predicted
  40–42; within the resolution of a 2-bit ladder.
* $F$ terms before cancellation: fail below 33 bits for **all three cases**,
  $\kappa$-independent. Predicted 36. The discrimination against $u$ — one
  threshold moves with $\kappa$, the other does not — worked as designed.

One rule the experiment forced rather than confirmed: below the cancellation
threshold the solver's *own* stopping test fires on an exactly-zero rounded
residual. Convergence must be judged by an independent fp64 residual, or the
kernel reports false convergence as fast convergence. This is the design
principle of mixed-precision iterative refinement rediscovered from the failure
side.

## 2. What the evaluator applies a configuration to

PETSc's precision is a configure-time typedef (`PetscReal` is one of `float`,
`double`, `__float128`, `__fp16`); no `Mat` or `PC` carries its own. A
configuration therefore cannot be applied to PETSc at all except as a uniform
rebuild, and the search does hundreds of probes. This rules out the obvious
design and forces a factorization:

$$\text{time}(\text{config}) \;=\; \underbrace{\text{iterations}(\text{config})}_{\text{tier 1: numpy search}} \;\times\; \underbrace{\text{cost per iteration}(\text{storage formats})}_{\text{tier 2: measured on hardware}}$$

* **Tier 1** is what parsimony already does: search over an algorithmic
  transcription with the iteration-budget criterion. Mixed configurations are
  free here. Output: a Pareto set of (bits, iterations) at several budget slacks.
* **Tier 2** measures cost per iteration, which for a Krylov or Newton–Krylov
  method is dominated by the sparse matrix–vector product and is a function of
  storage formats only. This needs no mixed-precision PETSc. It needs one
  templated SpMV kernel, parametrized on (matrix format, vector format), timed
  on the target hardware, *including* the conversion cost — the very thing that
  erased the 2013 gains on CPUs and is nearly free in-register on a GPU.

The factorization is validated at the one point where both sides exist: uniform
fp32 versus uniform fp64, where PETSc itself can be built and timed end to end
(`--with-precision=single`) and compared with iterations × measured cost.

The search does **not** minimize modeled time directly. A "faster than baseline"
predicate is not monotone in the set of high-precision variables (returning a
variable to fp64 can flip it false), which is the assumption delta debugging
needs; Precimonious lived with the resulting local minima. Instead: sweep the
iteration slack, obtaining the Pareto curve of bits against iterations; then
apply the cost model afterwards. Time is a report, not a search objective.

### The GPU job

This is the only place allocation is spent, and it is the paper's thesis
measured rather than argued. One Kokkos kernel, `spmv<MatT, VecT>`, CSR, timed
for (MatT, VecT) ∈ {fp64, fp32, bf16, fp16}² on a few matrices spanning
nnz/row and size, on one A100 and one CPU node. Report GB/s and time per SpMV.
Expected: on the GPU, time tracks bytes moved almost exactly; on the CPU,
conversion overhead makes mixed (fp32 matrix, fp64 vector) slower than the byte
count predicts — which is the 2013 story reproduced on modern silicon, and
belongs in the paper either way. Budget: well under one node-hour.

## 3. What makes the performance claim honest

**The optimization gap mostly does not apply at this granularity.** The gap
(tune at `-O0`, ship at `-O3`) bites tools whose variables are compiler
temporaries that optimization deletes or keeps in registers. Our variables are
algorithmic quantities — a stored matrix, a residual vector — that survive
optimization by construction because they live in memory. This is a genuine
advantage of operation-level granularity and should be stated.

**What the emulator does and does not model.** Parsimony's numpy evaluator
rounds *stored* quantities: the result of each named computation, at the point
it is named. It does not model arithmetic performed in reduced precision —
accumulating a dot product in bf16, or evaluating a residual's terms in fp16
before they cancel. Two readings follow:

* For *storage* formats — the matrix, the vectors, everything bandwidth-bound —
  the model is exact, and these are the findings with performance consequence.
* For *scalars* the CG study found tolerant to 8 bits ($\alpha$, $\beta$,
  $\rho$), the honest statement is "these need not be *accurate*", not "these
  may be *computed* in 8-bit arithmetic". They will be computed in registers at
  fp32 or fp64 regardless, at no cost. The finding is about robustness of the
  iteration to their error, and has no performance content of its own.
* The stored-vs-evaluated distinction for $F$ is the sharpest case, and the
  Bratu kernel exposes it deliberately: `fine_residual=True` rounds the
  Laplacian term and the exponential term separately before they are
  subtracted, so the cancellation is modeled. The prediction table above says
  the two readings differ by a factor $\kappa$ in required bits. If they do,
  that is the clearest demonstration available that storage precision and
  arithmetic precision are different questions, and that a tool answering one
  must not be read as answering the other.

**What is claimed, and what would falsify it.** The claim is that a
Newton–Krylov solver admits reduced-precision *storage* of its rate-limiting
quantities at an iteration cost predictable from $\kappa$, and that on
bandwidth-bound hardware this storage reduction is a measured, not modeled,
saving. It is falsified if the measured knee does not track $\log_2\kappa$, or
if tier 2 shows conversion overhead eating the bandwidth saving on the GPU as
it did on 2013 CPUs. Either outcome is reportable.

## What remains mechanical after this

* Implement the `Criterion` abstraction (done alongside this document) and the
  Bratu kernel; run the virtual-width sweep; check the knee against
  $\log_2 \kappa$.
* Reuse the CG kernel as the inner solve for a Newton–Krylov variant; the
  variables compose.
* Write and time the Kokkos SpMV kernel on Perlmutter.
* Build PETSc `--with-precision=single` and `double`, time SNES ex5 end to end,
  and check the factorization at the uniform-fp32 point.
* Fold results into the paper's §6 and remove the last `\pending`.
