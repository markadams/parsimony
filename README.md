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
bf16/fp8 formats, and bandwidth-bound kernels where halving a matrix's storage is
close to a 2× win. The search algorithms deserve another look precisely because what
they were searching *for* is now worth finding.

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

## License

BSD-3-Clause.
