# parsimony

Find the fewest bits a computation actually needs.

`parsimony` searches for the lowest-precision assignment of a program's floating-point
variables that still meets an accuracy target — and tells you how much that search
bought over simply lowering everything uniformly.

```console
$ parsimony tune examples/mpow.py:kernel --tol 1e-8 --text
variables      11
tolerance      1e-08
uniform floor  fp64  (3 probes)
mixed          {'fp64': 10, 'fp8e5m2': 1}  (350 probes)
mixed error    0.000e+00
storage        704 -> 648 bits  (8.0% smaller)
```

## Why now

This is a successor to [Precimonious][precimonious] (Rubio-González, Nguyen, Nguyen,
Demmel, Kahan, Sen, Bailey, Iancu, Hough — SC'13) and its [Blame Analysis][blame]
follow-on (ICSE'16), from the Berkeley/LBNL CORVETTE project.

Those tools were right about the method and unlucky about the hardware. Blame
Analysis Table 5 reports the speedup after tuning across ten programs and four error
thresholds, and for half of them the answer is **0.0% at every threshold** — on a
2013 CPU, fp64→fp32 bought a little vectorization and some cache, minus the cost of
inserted casts. The tools frequently concluded the original program was fastest.

The objective function has changed. Reduced precision now targets tensor cores,
bf16/fp8 formats, and bandwidth-bound kernels where narrowing a matrix's storage
cuts memory traffic. The search algorithms deserve another look precisely because
what they were searching *for* is now worth finding.

How much it cuts depends on the sparse data structure, not just the format. In
block CSR with block size `b`, one SpMV moves `r_b*(b^2*w + 4) + 4 + b*(v + 8)`
bytes per block row — narrowing the value width `w` leaves the column indices,
the row pointer, and the vector traffic alone. A block carries one index for
`b^2` values, so the saving is bounded by `(8 + 4/b^2)/(w + 4/b^2)`:

| block size | fp32 | fp16 |
|---|---|---|
| 1 (AIJ) | 1.50× | 2.00× |
| 3 (elasticity, BAIJ) | 1.90× | 3.45× |

So scalar CSR **cannot** halve the time — the 4-byte index per nonzero caps it,
and a 7-point stencil realizes only 1.36×. Blocking lifts the cap: for an
`ex56`-shaped elasticity matrix (`bs=3`, 27 blocks/row), BAIJ gives 1.85× and
3.24× where AIJ gives 1.48× and 1.96×, and BAIJ at fp32 moves **2.62× less than
AIJ at fp64**. Reduced precision and blocking are complementary. See
[`casestudies/spmv`](casestudies/spmv), whose host-only test asserts the
derivation.

## Two plugin axes

Prior tools fused these together, which is why each is welded to one measurement
technology. `parsimony` keeps them orthogonal:

| Axis | Question | Implementations |
|---|---|---|
| **Evaluator** | Given a precision configuration, how accurate is the program? | `numpy` (built in, zero deps) · verificarlo · verrou · native rebuild *(planned)* |
| **Search** | How do I explore the configuration space? | `UniformBisect` · `PrecisionDDMin` · blame-style attribution *(planned)* |

The split lets a cheap evaluator drive an expensive search during development, then
an expensive evaluator confirm the winner once. It also makes the search testable
against a synthetic oracle, independent of any floating-point emulation — see
`tests/test_ddmin.py`.

## Formats

IEEE fp64/fp32/fp16, plus bf16, tf32 and both fp8 encodings (e4m3, e5m2), plus
*virtual* widths — a float64 truncated to k significand bits, which is how you ask
"how many bits does this value need?" without committing to a storage format:

```console
$ parsimony tune examples/mpow.py:kernel --tol 1e-8 --ladder v53,v40,v30,v24,v16 --text
uniform floor  v30
```

Emulation is checked against hardware: `round_to(x, FP32)` is bit-identical to
`np.float32(x)` over random samples, including subnormals (`tests/test_formats.py`).

## Writing a kernel

A kernel is an ordinary function taking a recorder. Every value passed through it
becomes a tunable variable:

```python
def kernel(P):
    a   = P("a", 1.84089642)
    acc = P("acc", a * a)
    return {"res": acc}
```

This is deliberately the same shape as verificarlo's `vfc_probe`, so a kernel
prototyped here ports to an instrumented C build without rethinking the variables.

## Acceptance criteria

What "passes" means is pluggable (`parsimony.criteria`), and for iterative methods
it is not a detail. Relative error against the fp64 run is right for a function
that computes a value; for a solver it measures bit-reproduction of a rounding
trajectory near convergence, and for a Jacobian or preconditioner — which cannot
change the fixed point — it measures nothing. `IterationBudget` holds the
solver's own residual tolerance fixed and counts iterations instead:

```python
tune(kernel, cases=CASES, criterion=IterationBudget(slack=0.25))
```

## Case studies

- [`casestudies/nas_cg`](casestudies/nas_cg) — conjugate gradient. The scalars
  that steer the iteration tolerate bf16 and below; the operator and state vectors
  set accuracy, with fp32 operator storage sufficient to 1e-6. Three experimental
  design errors caught and documented.
- [`casestudies/newton`](casestudies/newton) — Bratu Newton. Design written first
  (`DESIGN.md`), then checked. fp32 Jacobian storage is free, fp16 costs a third
  more iterations, and below ~10 bits Newton degrades to Picard iteration rather
  than diverging. The residual stores in 16 bits but cannot be *evaluated* in
  fewer than 33 — storage and arithmetic precision are different questions.

## Status

The numpy evaluator, prefilter, both searches and both criteria work and are
tested (77 tests). What the numpy evaluator models is **storage** rounding — the
value of each named quantity at the point it is named — not arithmetic performed
in reduced precision; read every result with that in mind. Not yet built:

- **Evaluators that measure time.** Everything here counts iterations or bits. The
  design (`casestudies/newton/DESIGN.md`) factorizes time as iterations × cost per
  iteration and specifies the second factor as a templated SpMV kernel timed on
  hardware, since PETSc's precision is a configure-time typedef and a
  per-configuration rebuild is not an option.
- **Coarse-grained blame.** Attribution at the level of library operations rather
  than LLVM instructions — the instruction-level version cost 50× runtime and
  exhausted 256 GB on a NAS benchmark at class S.

## Prior and parallel work

- [Precimonious][precimonious] (SC'13) — ddmin over type configurations. BSD-3-Clause,
  © 2013 UC Berkeley. LLVM 3.0; unmaintained. No code is reproduced here; the
  algorithms are reimplemented from their published descriptions.
- [Blame Analysis][blame] (ICSE'16) — single-pass shadow execution, used as a prefilter.
- [PROMISE with customized precisions][promise] (Chen, Hilaire, Jézéquel, 2026) —
  the parallel revival, via CADNA and FloatX. Same realization, different lineage.
- [Verificarlo][verificarlo] — Monte Carlo arithmetic and variable precision by LLVM
  instrumentation; the intended heavyweight evaluator.

[precimonious]: https://dl.acm.org/doi/10.1145/2503210.2503296
[blame]: https://web.cs.ucdavis.edu/~rubio/includes/icse16.pdf
[promise]: https://arxiv.org/abs/2606.08339
[verificarlo]: https://github.com/verificarlo/verificarlo

## Install

```console
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Paper

`paper/parsimony.tex` — *Parsimony: Revisiting Search-Based Floating-Point
Precision Tuning for Reduced-Precision Hardware*. Build with `make` in `paper/`
(pdflatex + bibtex).

## Data availability

Every number in the paper traces to a run in this repository. Each case study
directory holds the batch script that produced its results next to the unedited
`-log_view` output:

| directory | what it measures | raw data |
|---|---|---|
| [`casestudies/nas_cg`](casestudies/nas_cg) | CG precision assignment | `results.json` |
| [`casestudies/newton`](casestudies/newton) | Bratu Newton, per-variable bit sweep | `sweep.json`, `search.json` |
| [`casestudies/spmv`](casestudies/spmv) | A100 SpMV by storage format | `results-a100.txt` |
| [`casestudies/e2e`](casestudies/e2e) | end-to-end PETSc elasticity solve | `ex56k-*/`, `scale-*/` |

The end-to-end measurements use a PETSc development fork, not a release: upstream
PETSc has neither the BAIJ-Kokkos matrix type nor the Kokkos FE template path, so
the blocked and device-assembly arms are not reproducible against a released
version. Measurements were taken on NVIDIA A100 GPUs at NERSC.

## Use of AI

Anthropic's Claude models, operating agentically through Claude Code, were used
under the author's direction for background research and exploration, analysis,
code writing, debugging, and experiment scripting, and for drafting of the
manuscript. The author made all design and experimental decisions, verified all
results, and takes full responsibility for the content.

Two corrections are recorded in the paper's *Use of AI* section because they bear
on using such tools for measurement: a GPU profile mis-attributed by asynchronous
kernel launches, caught because the implied bandwidth was physically impossible;
and a single-GPU blocked-format deficit first reported as a defect before being
identified as documented, expected behaviour at that scale. Both were caught by
checking a number against a physical bound or against prior work.

## License

BSD-3-Clause.
