"""Conjugate gradient, transcribed for precision tuning.

This follows the inner loop of NAS `cg` -- the benchmark Precimonious and Blame
Analysis both tuned -- keeping the C variable names (`rho`, `rho0`, `alpha`,
`beta`, `d`, `q`, `z`, `p`, `r`) so configurations are readable against the
published ones.

Two honest differences from the benchmark.  The operator here is a
variable-coefficient diffusion stencil rather than NAS' randomly generated sparse
matrix, because its conditioning is controllable, which matters more for a
precision study than reproducing the generator.  And the tuned variable set is
the algorithm's working set (10 variables), not the 32 floating-point variables
the C program declares, so variable *counts* are not comparable across the two --
only which quantities need precision and which do not.

The coefficient field is deliberately not constant.  A constant-coefficient
five-point Laplacian has entries 4 and -1, both exactly representable in every
format down to 8 bits, so `A` appears to need no precision at all -- an artifact
of the operator, not a finding about storage.  Generic entries are required for
the matrix-storage question to mean anything.

Iteration count is fixed rather than convergence-tested, as in NAS `cg`.  A
convergence test would let the count vary with precision, confounding "this
format is less accurate" with "this format took more iterations".

The count is calibrated per case: it is the number of iterations the fp64 run
needs to reach a relative residual of 1e-12, and is then held constant across
every precision configuration.  Calibrating matters.  Running a fixed large count
for every case leaves the well-conditioned ones iterating in the noise floor,
where `rho = ||r||^2` underflows at low precision and the run returns NaN -- so
the search ends up measuring underflow in dead iterations rather than solver
accuracy.

`A` is rounded once, outside the loop: that models the matrix being *stored* in a
given format, which is the decision that matters for a bandwidth-bound solver.
"""

from __future__ import annotations

import numpy as np



def diffusion_2d(k: int, contrast: float = 8.0) -> np.ndarray:
    """Variable-coefficient diffusion on a k x k grid, dense (k^2, k^2) and SPD.

    Face coefficients are harmonic means of a smooth cell-centred field, giving
    generic (non-dyadic) matrix entries.  `contrast` sets the ratio between the
    largest and smallest coefficient, and with it the conditioning.
    """
    x = (np.arange(k) + 0.5) / k
    kappa = np.exp(np.log(contrast) * 0.5 *
                   np.sin(3.0 * np.pi * x)[:, None] * np.cos(5.0 * np.pi * x)[None, :])

    n = k * k
    A = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            row = i * k + j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < k and 0 <= jj < k:
                    face = 2.0 / (1.0 / kappa[i, j] + 1.0 / kappa[ii, jj])
                    A[row, ii * k + jj] = -face
                    A[row, row] += face
                else:
                    A[row, row] += kappa[i, j]  # Dirichlet boundary
    return A


def _iterations_to_converge(A: np.ndarray, b: np.ndarray,
                            rtol: float = 1e-12, cap: int = 500) -> int:
    """Iterations plain fp64 CG needs to reach a relative residual of `rtol`."""
    x = np.zeros_like(b)
    r = b - A @ x
    p, rho, target = r.copy(), r @ r, rtol**2 * (b @ b)
    for n in range(1, cap + 1):
        q = A @ p
        alpha = rho / (p @ q)
        x += alpha * p
        r -= alpha * q
        rho0, rho = rho, r @ r
        if rho <= target:
            return n
        p = r + (rho / rho0) * p
    return cap


def make_case(k: int):
    """Solve A x = b with b chosen so the exact solution is all ones.

    A uniform solution keeps every component O(1), so a componentwise relative
    error is well conditioned and no single near-zero entry dominates the metric.
    """
    A = diffusion_2d(k)
    b = A @ np.ones(A.shape[0])
    return A, b, _iterations_to_converge(A, b)


#: Three grid sizes, so the search cannot overfit one conditioning.
CASES = (make_case(8), make_case(12), make_case(16))


def kernel(P, case=None):
    if case is None:
        case = CASES[0]
    A0, b, niter = case

    A = P("A", A0)  # storage format of the operator
    x = np.zeros(b.shape[0])
    r = P("r", b - A @ x)
    p = P("p", r.copy())
    z = P("z", x.copy())
    rho = P("rho", float(r @ r))

    for _ in range(niter):
        q = P("q", A @ p)
        d = P("d", float(p @ q))
        alpha = P("alpha", rho / d)
        z = P("z", z + alpha * p)
        rho0 = P("rho0", rho)
        r = P("r", r - alpha * q)
        rho = P("rho", float(r @ r))
        beta = P("beta", rho / rho0)
        p = P("p", r + beta * p)

    return {"z": z}
