# Local patches to Grid

Against `paboyle/Grid` develop `865c338dd2594d1e0bd6b249a13bcbd5e0b77b06`.
Both apply cleanly to that commit (`git apply --check`), and both are
candidates to send upstream (Q10 and Q11 in `../share/QUESTIONS-FOR-PETER.md`).

| patch | why |
|---|---|
| `0001-perfcount-rdtsc-shim.patch` | Build break with cudatoolkit 13.2: Grid's no-op `__rdtsc` shim collides with the real declaration (`cannot overload functions distinguished by return type alone`). Uses are host-only, so include `x86intrin.h` instead. |
| `0002-twolevelcgmrhs-expose-iterations.patch` | `TwoLevelCGmrhs` prints its iteration count and true residual but does not expose them. Adds `IterationsToComplete`, `TrueResidual`, `ConvergedFlag`, as `ConjugateGradient` has. |

```bash
cd Grid && git apply /path/to/grid-patches/000*.patch
```

The tree on Perlmutter that produced the results differs from `0002` only in
trailing whitespace an editor stripped; `0002` is the clean version.
