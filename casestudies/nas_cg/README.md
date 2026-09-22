# Validation: NAS `cg`

The credibility anchor. NAS conjugate gradient is in Precimonious' own benchmark
set, so its published configuration and speedup are directly comparable — if the
clean-room `ddmin` here cannot reproduce them, nothing built on top is trustworthy.

## Published numbers to reproduce

From Blame Analysis (ICSE'16), Tables 4 and 5. The `Initial` column gives 32 double
and 0 float variables declared in the program.

| Error threshold | Blame (D/F) | Blame + Precimonious (D/F) | Precimonious (D/F) | Speedup (B+P / P) |
|---|---|---|---|---|
| 1e-4  | 26 / 6 | 2 / 30  | 2 / 30  | 7.7% / 7.7% |
| 1e-6  | 28 / 4 | 13 / 19 | 13 / 19 | 7.9% / 7.9% |
| 1e-8  | 28 / 4 | 16 / 16 | 12 / 20 | 7.9% / 7.4% |
| 1e-10 | 28 / 4 | 16 / 16 | 16 / 16 | 7.9% / 7.9% |

`cg` is one of only five programs in the study where tuning produced any speedup at
all, which is exactly why it is the right validation target: the accuracy result and
the performance result are both non-trivial.

## What to check, in order

1. **Configuration agreement.** Does `PrecisionDDMin` find a comparable double/float
   split at each threshold? Exact agreement is not expected — the variable set at
   LLVM bitcode level differs from a Python or C source-level set, and the paper
   tuned at `-O0` because optimization deletes variables. Agreement in *count and
   character* (which variables survive in double) is the bar.
2. **Search cost.** The paper reports Precimonious taking hours on programs with
   fewer than 50 variables. Record probe counts here; they are the baseline the
   blame prefilter has to beat.
3. **Threshold monotonicity.** A tighter threshold must never admit more lowering.
   Table 4 satisfies this; a search that violates it has a monotonicity bug.

## Status

Not started. Needs an evaluator that can run the benchmark — either a numpy
transcription of the `cg` kernel (fast to write, loses the performance half of the
result) or a verificarlo/native evaluator (slower to build, but the only way to
check the speedup column).

The numpy transcription is the right first move: it validates the *search* against
published configurations without needing a compiler, which is the half that matters
for trusting everything else.
