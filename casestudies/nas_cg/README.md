# Case study: conjugate gradient

The credibility anchor. NAS `cg` is in Precimonious' own benchmark set and is one
of only five programs in the Blame Analysis study that showed any speedup, so it
exercises both halves of the dual objective.

Run it with `python run.py`; results land in `results.json`.

## What is and is not comparable

The kernel here (`examples/nas_cg.py`) transcribes the CG inner loop with the C
variable names preserved, but it is **not** the NAS benchmark:

| | NAS `cg` | here |
|---|---|---|
| operator | randomly generated sparse | variable-coefficient diffusion, 3 grids |
| tuned variables | 32 declared in the C program | 10 working-set quantities |
| formats | fp32 / fp64 / fp80 | fp64 down to two 8-bit encodings |
| iteration count | fixed 25 | calibrated per case to fp64 convergence |

So **variable counts are not comparable**. What is comparable is the character of
the answer: which quantities need precision and which do not.

## Results

Full search (no prefilter), relative error against the fp64 run, three cases:

| tolerance | bits | A | z | r | q | p | rho | rho0 | d | alpha | beta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4  | 208 | fp32 | fp32 | fp32 | fp32 | bf16 | bf16 | bf16 | bf16 | fp8e5m2 | fp8e5m2 |
| 1e-6  | 224 | fp32 | fp32 | fp32 | fp32 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 |
| 1e-8  | 419 | fp64 | fp64 | fp64 | fp64 | fp32 | fp32 | tf32 | fp32 | fp32 | fp16 |
| 1e-10 | 640 | fp64 | fp64 | fp64 | fp64 | fp64 | fp64 | fp64 | fp64 | fp64 | fp64 |

Storage is monotone in tolerance (208 → 224 → 419 → 640 bits): tightening never
admits more lowering, which is the basic consistency check on the search.

Three things stand out.

**The scalars are cheap.** `alpha`, `beta`, `rho`, `rho0` and `d` sit at or below
bf16 wherever the vectors are at fp32, and `alpha`/`beta` reach 8 bits at 1e-4.
This is the CG form of a familiar fact: the step length and conjugacy coefficient
only steer the iteration, and an inaccurate step is corrected by the next residual.
The quantities that *carry state* — the matrix and the vectors — set the achievable
accuracy.

**The search direction is cheaper than the residual.** `p` is consistently a rung
or two below `r`, `z` and `q`. Same mechanism: `p` sets a direction, while `r` and
`z` accumulate the answer.

**The matrix tracks the state vectors**, at fp32 for tolerances of 1e-6 and looser
and fp64 beyond. Since sparse matrix-vector products are bandwidth-bound, this is
the storage decision with the largest performance consequence, and the answer is
not trivially "fp64 everywhere".

### Prefilter trade

| tolerance | probes (full → prefiltered) | speedup | bits (full → prefiltered) |
|---|---|---|---|
| 1e-4  | 214 → 35 | 6.1× | 208 → 224 (+7.7%) |
| 1e-6  | 229 → 36 | 6.4× | 224 → 227 (+1.3%) |
| 1e-8  | 306 → 27 | 11.3× | 419 → 448 (+6.9%) |
| 1e-10 | 294 → 20 | 14.7× | 640 → 640 (0%) |

6–15× fewer evaluations for at most 8% more bits. For comparison, Blame Analysis
reported its prefilter making the combined analysis 9× faster on average, by a far
more expensive mechanism — shadow execution at instruction granularity, costing up
to 50× runtime.

## Comparison with the published result

Blame Analysis Table 4 for `cg`, as a fraction of variables lowered below double:

| tolerance | published (B+P) | here |
|---|---|---|
| 1e-4  | 30/32 = 94% | 10/10 = 100% |
| 1e-6  | 19/32 = 59% | 10/10 = 100% |
| 1e-8  | 16/32 = 50% | 6/10 = 60% |
| 1e-10 | 16/32 = 50% | 0/10 = 0% |

Partial agreement: both lower nearly everything at 1e-4 and roughly half at 1e-8.
The divergence at 1e-10 is a property of the acceptance criterion, not of the
search — see below.

## Threats to validity found while building this

Three experimental-design errors were caught here. Each produced a plausible-looking
result that was meaningless, and each is cheap to make.

**A constant-coefficient operator makes the matrix look free.** The first version
used a five-point Laplacian, whose entries are 4 and −1 — both exactly representable
in every format down to 8 bits. `A` came out at fp8 at *every* tolerance including
1e-10, which reads as a dramatic finding and is actually the statement that
rounding a no-op changes nothing. Generic matrix entries are required for the
storage question to mean anything.

**A fixed iteration count leaves well-conditioned cases in the noise floor.**
Running 120 iterations on problems that converge in 31–76 meant the tail iterations
operated on noise-level residuals, where `rho = ||r||²` underflows at low precision
and the run returns NaN. The search was then partly measuring underflow in dead
iterations. Counts are now calibrated per case from the fp64 run.

**One input lets the search cheat.** See `examples/mpow.py`: with a single input, a
tolerance can be met by coarsening an *output* until it quantizes onto a lattice
point near the reference. Three cases are used here for that reason.

## A limitation of the fp64-reference criterion

At 1e-10 every variable comes back fp64, and this persists after tightening the
reference's own convergence to 7e-13 — so it is not a convergence artifact.

Reduced-precision CG does not converge to the *same iterate* as fp64 CG; it
converges to a nearby but different point. Demanding agreement with the fp64 answer
to ten digits therefore demands reproducing a particular rounding trajectory, not
merely solving the problem well. As the tolerance approaches the solver's own
convergence level, the Precimonious-style acceptance criterion stops measuring
solution quality and starts measuring bit-reproduction.

For iterative solvers, a criterion phrased on the *residual*, or on error against
the exact solution, would be better posed than one phrased on agreement with a
full-precision run. That is a change to the acceptance criterion rather than to the
search, and it applies to this whole family of tools.
