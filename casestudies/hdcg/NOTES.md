# HDCG precision study: running notes

Companion to PLAN.md. Records what has been built and verified, what the
machine actually looks like, and what is blocking. Updated 2026-09-27.

## Status

Grid is built on Perlmutter and the rounding machinery runs on GPUs against
real Grid data. **The HDCG driver itself is blocked** by a Grid-side problem
(below), so the first measurement was taken on the coarsening path that does
run: a full bit-ladder for the coarse Dirac stencil, free field, 2 cases. See
"First results" below.

| plan item | state |
|---|---|
| Day 1 — Grid built on Perlmutter (CUDA, `865c338d`) | **done** |
| Day 1 — baseline HDCG solve | **blocked**, `MultiGeneralCoarsenedMatrix` segfaults |
| Day 2 — `RoundField` + unit test vs `round_to` | **done and passing** |
| Day 2 — `grid_eval.py` evaluator | **done**, tested against a stand-in driver |
| Day 2 — instrumented HDCG driver | **built**, cannot get past setup |
| — coarse-stencil bit ladder (substitute) | **done**, 12 formats × 2 cases |
| Days 3–4 — HDCG search runs | blocked on the above |
| Day 5 — Fable analysis pass | partial data available |

## The build that worked

Mark cloned Grid; the rest is reproducible from here.

```bash
git clone -b develop https://github.com/paboyle/Grid.git /pscratch/sd/m/madams/Grid
cd /pscratch/sd/m/madams/Grid && ./bootstrap.sh     # fetches Eigen 3.4.0
export CRAY_ACCEL_TARGET=nvidia80
mkdir build && cd build
../configure --enable-comms=mpi --enable-simd=GPU --enable-shm=nvlink \
  --enable-gen-simd-width=64 --enable-accelerator=cuda \
  --disable-fermion-reps --disable-unified --disable-gparity \
  --with-mpfr=/pscratch/sd/m/madams/sw/mpfr \
  CXX=nvcc LDFLAGS="-cudart shared -Xlinker -rpath -Xlinker /pscratch/sd/m/madams/sw/mpfr/lib" \
  CXXFLAGS="-ccbin CC -gencode arch=compute_80,code=sm_80 -std=c++17 -cudart shared"
make -j8          # ~35 min on a login node; libGrid.a is 181 MB
```

Four things the plan got wrong or left open, each settled by doing it:

* **mpfr is a hard requirement, not a warning.** Grid's configure ends with
  `configure: error: MPFR library not found`. Perlmutter has `libmpfr.so.6` but
  no header and no `.so` symlink. Rather than build it, copy NERSC's own: mpfr
  4.2.2 with headers lives in `/global/common/software/nersc/pe/conda-envs/
  latest/python-3.13/nersc-python/`. Copy `mpfr.h`, `mpf2mpfr.h` and
  `libmpfr.so.6.2.2` into `/pscratch/sd/m/madams/sw/mpfr/{include,lib}` with the
  usual symlinks. Isolating it matters twice over: nothing else from that conda
  env reaches the link line, and the copy survives NERSC rotating the `latest`
  symlink. It then resolves against the *system* gmp, same soname, no conflict.
* **cudatoolkit/13.2 with gcc-native/14 is correct; the plan's 12.4 fallback is
  wrong.** 12.4 refuses gcc-14 outright (`unsupported GNU version! gcc versions
  later than 13`). Confirmed by compiling Grid's `__rdtsc` shim standalone under
  all four combinations.
* **`-Wl,` does not reach nvcc.** `nvcc fatal : Unknown option
  '-Wl,-rpath,...'`. Use `-Xlinker -rpath -Xlinker <dir>`.
* **A login-node build is fine** at `-j8`. An earlier build looked dead because
  `pgrep nvcc` returned zero between compiler invocations; use the log's mtime
  as the liveness signal, not the process table. (One healthy build was killed
  on that bad reading.)

### Two local patches to Grid

Both are in `/pscratch/sd/m/madams/Grid`, both small, both candidates to send
upstream. `PerfCount.h.orig` is kept beside the first.

1. **`Grid/perfmon/PerfCount.h`** — build break on current NERSC defaults.
   Under `GRID_CUDA` Grid defines no-op `__rdtsc`/`__rdpmc` shims to keep
   `x86intrin.h` away from nvcc, but CUDA 13.2's header chain now declares the
   real `__rdtsc` first, so the shim collides: *cannot overload functions
   distinguished by return type alone* (`unsigned long long` against
   `uint64_t`). The two uses are in `Stat.cc`, host-side performance counters,
   never in a device kernel — so the shim is replaced by
   `#include <x86intrin.h>` and the genuine intrinsic is used.
2. **`Grid/algorithms/iterative/AdefMrhs.h`** — `TwoLevelCGmrhs` prints its
   iteration count and true residual but exposes neither. Added
   `IterationsToComplete`, `TrueResidual` and `ConvergedFlag`, filled on both
   the converged and non-converged exits, exactly as `ConjugateGradient`
   already does. Without this the driver would have to scrape stdout.

## Open failure: segfault in CoarsenOperator

Every run dies at the same point, with rounding **off** — so this is not the
instrumentation:

```
Building MultiRHS Coarse operator
Grid : Message : NonLocalStencilGeometry has 81 terms in stencil
srun: error: tasks 0-3: Segmentation fault
```

The crash is immediately after the `NonLocalStencilGeometry` inside
`MultiGeneralCoarsenedMatrix`'s constructor prints — the second time that
message appears in a run, the first being the driver's own `geom`.

**What has been ruled out, each by measurement:**

| suspect | test | result |
|---|---|---|
| memory pool / comms flags | with and without `--device-mem 16000 --shm 2048 --comms-sequential --shm-mpi 0` | crashes either way |
| MPI decomposition | `--mpi` 1.1.1.1, 1.1.1.2, 1.1.1.4 | **all three crash, including one rank** |
| subspace quality | Chebyshev order 500 and 20 | crashes either way |
| the rounding hooks | every run above has rounding off | not involved |

Two dead ends worth recording so they are not re-walked. First, a halo-depth
argument — coarse lattice 4·4·4·8, `--mpi 1.1.1.4` giving a T extent of 2
against a depth-3 stencil — is wrong: it survives `--mpi 1.1.1.2` (extents
4·4·4·4) and single-rank. Second, **`exit=0` from `srun` does not mean the run
succeeded** once `--debug-signals` is on: Grid's handler catches SIGSEGV,
prints a backtrace and lets the step exit clean. Check for `Caught signal 11`,
not the exit status.

The backtrace has no symbols (built `-O3`, no `-g`), so it gives only the
crash address.

**Diagnosis: device memory.** `tests/debug/Test_general_coarse.cc` — Grid's own
unmodified test, same class, its own hot configuration, no gauge file — fails
at the *identical* point and says so properly:

```
Grid : Message : Building little Dirac operator
Grid : Message : NonLocalStencilGeometry has 81 terms in stencil
 cudaMalloc failed for 8153726976 out of memory
```

So the segfault in our driver is an allocation failure that Grid happens not to
check on that path, and the build, the hooks and the decomposition are all
exonerated.

It does not scale away. Sweeping the control over `--device-mem` ∈ {2000, 8000,
32000} × ranks ∈ {1, 4}: **every 1-rank run fails, every 4-rank run passes**,
and the cache size makes no difference either way. But our driver still fails
at 4 ranks *and* at 8 ranks on two nodes, at every cache size.

That is expected once the footprint is written down. The coarse grid is 512
sites; the stencil has 81 points and pads a depth-3 halo. Local coarse extents
of 4·4·4·2 pad to 10·10·10·8, so 8000 padded sites back 128 real ones, and each
site carries a 40×40 ComplexD block — 25.6 KB. That is ~16 GB per rank of
padded coarse operator, and because the halo dominates, doubling the ranks
barely reduces it. Adding GPUs is the wrong lever.

The levers that would work, in order of how much they buy:

* **`nbasis`** — the coarse block is nbasis², so 40 → 20 is a 4× cut. It is a
  template parameter, so this costs a rebuild and nothing else.
* **lattice volume** — 16³×32 → 16³×16 halves the coarse sites.
* **ask Peter what he runs this on.** This is the stock configuration of his own
  test, unmodified, and it does not fit on four (or eight) A100s. That is worth
  asking at the 09-28 meeting rather than guessing, and it is a better question
  than anything on the current asks slide.

Still worth doing regardless: get a real configuration, and run the *stock*
`Test_general_coarse_hdcg` unmodified as the final control.

## First results: bits in Grid's coarse Dirac stencil

Since the HDCG driver is blocked (above), the measurement was moved to the
coarsening path that *does* run on develop: `Test_coarse_bits.cc`, built like
`Test_general_coarse.cc` — random aggregation, single-RHS
`GeneralCoarsenedMatrix`, no multi-RHS class. Two coarse operators are built
from one aggregation; one has its stored stencil rounded, the other is kept
exact. CG runs against the rounded one and the answer is judged against the
exact one, because the rounded operator converges accurately to its own,
different, solution.

**Preliminary and deliberately not physics**: free field (unit gauge), random
subspace, nbasis 20, Ls 8, 16³×32, 4 A100s, CG tolerance 1e-8. Every JSON
record carries all of those so a cheap run cannot later be read as a physics
one. Raw records in `results/`, job log in `results/sweep.log`.

| stored as | sig. bits | δ (rel. L2 change) | true_resid | CG iters |
|---|---|---|---|---|
| fp64 (`v53`) | 53 | 0 | 3.089e-09 | 5 |
| `v40` | 40 | 3.88e-13 | 3.089e-09 | 5 |
| `v30` | 30 | 3.97e-10 | 3.100e-09 | 5 |
| `v24` / **fp32** | 24 | 2.54e-08 | 1.834e-08 | 5 |
| `v20` | 20 | 4.07e-07 | 2.920e-07 | 5 |
| `v16` | 16 | 6.51e-06 | 4.654e-06 | 5 |
| `v12` | 12 | 1.04e-04 | 7.476e-05 | 5 |
| `v10` | 10 | 4.17e-04 | 3.024e-04 | 5 |
| `v8` / **bf16** | 8 | 1.78e-03 | 1.195e-03 | 5 |
| **fp16** | 11, 5 exp | **1.0** | 1.500e-04 | 5 |

Case 1 reproduces every row to three digits.

**1. The iteration count never moves.** Five iterations at every format, down
to eight significand bits. The coarse operator is rate-limiting, not
accuracy-limiting, and the Newton study's division (DESIGN.md §1) shows up
directly: a degraded operator changes *which* problem is solved, not how long
it takes to solve it. For the coarse operator's actual job in HDCG — sitting
inside a preconditioner, where the fixed point is set by the fine operator —
this column is the relevant one, and it says the stencil is very tolerant.

**2. The accuracy floor tracks the stored precision exactly**: δ ≈ 2^-(k+1.2)
over nine rungs spanning 3e-9 to 1.2e-3, and true_resid ≈ 0.7 δ once δ rises
above the solver tolerance. So *as a solve target* the stencil needs about 30
bits to preserve the fp64 answer at 1e-8 — which is simply log₂(1/tol) plus
margin, not a property of the operator.

**3. The emulation is confirmed against hardware semantics on real data.**
`fp32` reproduces `v24` and `bf16` reproduces `v8` — identical to every digit
in δ and true_resid, in both cases. `round_bits.h` was checked against
`formats.round_to` on the Mac; this is the same kernel agreeing with what the
named formats mean, on Grid coarse-operator data on an A100.

**4. fp16 fails on exponent range, not precision.** δ = 1.0 exactly means the
rounded stencil is identically zero: every entry underflows fp16's 5-bit
exponent. bf16, with *three fewer* significand bits but fp32's exponent range,
is fine. **Caveat before this is repeated anywhere**: the entries of this
particular operator are tiny (below ~6e-8) because of the free field and the
random subspace, so the fp16 result is partly a scaling artifact of the test.
The mechanism is real and is the reason bf16 exists; whether a physics coarse
operator's entries sit in fp16's range is a separate question that needs the
real configuration. Note also that true_resid for fp16 (1.5e-4) is *better*
than for bf16 — solving with a zeroed operator happens to land closer, which
is a good reminder that true_resid alone does not rank formats.

### Cost

Setup dominates and is the thing to optimise if this becomes a search:

| phase | cost |
|---|---|
| `CreateSubspaceChebyshev`, per basis vector (nbasis 40, Ls 16) | 4.8–5.0 s |
| `CoarsenOperator` (nbasis 20, Ls 8), per operator | ~27 s |
| whole setup per evaluation | 53 s |
| CG solve | 0.015–0.11 s |

The solve is four orders of magnitude cheaper than the setup, which settles
**open question 1** for this configuration: a search pays setup × evaluations
and nothing else, so caching the coarsened operator (or the subspace) is the
only optimisation worth making.

## What is built, and how far it is trusted

**`round_bits.h`** — storage rounding of a double to a virtual (`v40`) or named
(`bf16`) format. Grid-free and device-callable by construction, so that it can
be tested where Grid is not.

**Tested, hard** (`tests/test_hdcg_round.py`, 16 cases): the C++ agrees with
`parsimony.formats.round_to` *exactly* — bitwise, in hex-float, over 372
doubles spanning ordinary magnitudes, exact ties (where round-half-to-even is
the whole question), signed zero, and the exponent extremes where fp16 and
bf16 overflow and underflow. Checked for all seven virtual rungs and all six
named formats, plus directly against hardware `float32`/`float16` casts. This
matters because the Python side chooses the configuration and the C++ side
applies it: had they disagreed, the study would have reported the knee of two
different functions and nothing in the output would have said so.

**`round_field.h`** — the Grid `Lattice` wrapper. Now compiles under nvcc.
Three things it had to learn the hard way: `autoView` and `accelerator_for`
are macros and cannot be namespace-qualified; they expand to unqualified Grid
names, so the function body needs `using namespace Grid;`; and there is no
`Grid::uint64_t`. It rounds every double in a `vobj`, which is
per-real-component for a ComplexD field, and refuses single-precision fields
by name rather than by `sizeof` — a ComplexF field is 8 bytes and would pass a
naive divisibility check, then be reinterpreted as garbage. **Not yet
exercised on real data**: no run has reached a hook.

Each rounding returns the relative L2 change it caused. That is not
instrumentation for its own sake: a hook attached to the wrong object, or
attached after the value is consumed, yields a flat curve that reads exactly
like "this input is free". `grid_eval.py` raises `RoundingNoOp` on a zero
change where rounding was requested, so that failure stops the search instead
of being written up as a finding.

**`Test_hdcg_parsimony.cc`** — `Test_general_coarse_hdcg.cc` plus the three
hooks, the command line, and the JSON record. Builds clean. Two things worth
knowing:

* `--case 0` deliberately does **not** reseed, so it leaves RNG5 exactly where
  the unmodified driver leaves it. An all-`v53` run therefore reproduces the
  stock driver's iteration count exactly and the sanity gate is an equality.
  Cases 1, 2, ... reseed, which changes only the right hand sides.
* The stencil hook round-trips each of the 81 geometry points through
  `GetMatrix`/`SetMatrix`, which needs a `std::vector<CoarseMatrix>` of length
  `npoint` on the single-rhs coarse grid — about 266 MB per rank at nbasis 40.

Because the tree has no `Makefile.am` entry for it, it is compiled by hand
with the flags `make V=1` prints for the stock driver; see
`/pscratch/sd/m/madams/hdcg/build_driver.sh`. Running `scripts/filelist` and
regenerating the autotools would be the tidy route if this becomes permanent.

**`grid_eval.py`** — the Parsimony evaluator and the search that drives it.
Tested against a stand-in driver with planted knees (subspace 16, stencil 30,
evec 8 bits); `SoloSensitivity` + `PrecisionDDMin` recover all three exactly in
16 solves. Also pinned: the command line the real driver must parse, the JSON
it must emit, that a crashed driver is an error rather than a divergence, that
probes are memoized across a dead allocation, and that sweeping the slack for
the Pareto curve costs no extra GPU time.

## Still blocked

**`ckpoint_lat.4000` has not arrived** — the plan's one external input. Mark is
asking Joe for it or for his `GRID_LATTICE_DIR`. Until then the driver runs
`--gauge hot`, which validates the pipeline; its knees are not reportable and
any figure made from it must say so. The driver prints a warning on every such
run so this cannot be forgotten quietly.

## Open questions, to settle from data rather than argument

1. **Is the subspace worth caching?** See "Measured so far" — leaning yes,
   pending a solve time.
2. **Does the `stencil` knee land at or below 8 explicit bits?** That is the
   one result that would justify raising 16-bit *storage* of the coarse stencil
   with Peter, and it is the only gap left in his kernel work. Below that bar,
   say nothing.
3. Is `evec` the most tolerant of the three, as the mechanism predicts? If not,
   suspect the harness before the physics.

## How production Grid does mixed precision (read 2026-09-30)

Needed before designing the next study, which aims to match production.

* **The precision boundary is the whole preconditioner, not the smoother.**
  `MrhsMixedPrecPreconditioner` (`multigrid/MrhsMultiGrid.h:267`) converts the
  outer solver's fp64 residual to fp32, runs an `MrhsPreconditioner<FieldF>` —
  "the whole V-cycle" — and converts the correction back. Its comment: "The
  outer operator never sees fp32." So production keeps a separate fp32
  operator for the preconditioner, with fp32 data *and* fp32 arithmetic.
* **Reductions are hybrid.** `innerProduct` (`lattice/Lattice_reduction.h`)
  forms the per-site product in the field's own precision, then sums across
  sites with `sumD`, which accumulates in double. Grid's own comment on it:
  "This is in single precision and fails some tests."

So our storage-only model (fp64 everything) is **optimistic** against
production: it does the preconditioner's arithmetic, and its per-site
reductions, in fp64. To match production below fp32, run the preconditioner on
Grid's real single-precision types — genuine fp32 arithmetic and Grid's own
hybrid reductions, for free — and round the *fp32* operator data to k < 24
bits. That reproduces production exactly at fp32 and extends it to
bf16/fp16 storage. `RoundField` refuses float fields today and needs a float
variant.

## Standing caveat for every write-up

This measures **storage** rounding of preconditioner inputs with fp64
arithmetic throughout. It does not measure reduced-precision *arithmetic* —
not Peter's ComplexF coarse space, not the 16F/BF16 compute types. It
complements his fp32-vs-fp64 comparison with a per-variable knee; it does not
reproduce or contradict it. Limitations to carry into the paper text: 512
coarse sites (tiny), one 16³×32 configuration, storage not arithmetic.
