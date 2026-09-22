"""Newton on the Bratu problem, transcribed for precision tuning.

Solves -laplace(u) - lambda * exp(u) = 0 on the unit square with zero Dirichlet
data, by Newton's method with a direct linear solve -- PETSc's SNES ex5 in its
simplest form.  The direct solve is deliberate: it isolates the precision of the
*stored* Jacobian from the inexactness of an inner Krylov method.  A
Newton-Krylov variant composes this with the CG kernel.

Variables, and what the design predicts for each (casestudies/newton/DESIGN.md):

  J      the assembled Jacobian, rounded once per Newton step: the storage
         format of the operator.  Rate-limiting; needs ~log2(kappa) bits.
  delta  the Newton step as stored.  Rate-limiting.
  F      the residual as stored after fp64 evaluation.  Predicted benign:
         a relative perturbation of the right-hand side, contraction u_p.
  F.lap, F.exp
         with fine_residual=True, the two terms of F rounded *before* they are
         subtracted, so the cancellation is modeled.  Predicted accuracy-limiting:
         attainable residual floor ~ u_p * kappa.
  u      the iterate as stored.  Accuracy-limiting.

The kernel returns the iteration count and a convergence flag, for the
IterationBudget criterion, plus the final relative residual for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    k: int                    # interior grid points per side; n = k*k unknowns
    lam: float                # Bratu parameter; turning point is near 6.81
    rtol: float = 1e-10
    maxit: int = 60
    fine_residual: bool = False


def _grid(k: int):
    h = 1.0 / (k + 1)
    n = k * k
    L = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            row = i * k + j
            L[row, row] = 4.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < k and 0 <= jj < k:
                    L[row, ii * k + jj] = -1.0
    return L / (h * h)


def _residual(P, L, u, lam, fine):
    lap = L @ u
    ex = lam * np.exp(u)
    if fine:
        lap = P("F.lap", lap)
        ex = P("F.exp", ex)
    return lap - ex


def _jacobian(L, u, lam):
    return L - np.diag(lam * np.exp(u))


def _true_residual(L, u, lam):
    """The residual in fp64 from the stored iterate, outside any rounding."""
    return float(np.linalg.norm(L @ u - lam * np.exp(u)))


def newton(P, case: Case):
    """Newton with the solver's own stopping test, judged by an independent residual.

    The solver stops when *its* residual -- the one it computed, possibly from
    rounded terms -- meets the tolerance.  Whether it actually converged is
    decided afterwards from an fp64 residual of the returned iterate.  The two
    differ: near the solution the terms of F nearly cancel, and at low precision
    they can round to the same representable value, so the computed residual is
    exactly zero and the solver declares victory falsely.  A real solver has
    only its own residual to look at, which is why mixed-precision iterative
    refinement evaluates the residual in high precision.  The factor of 2 on the
    check absorbs the honest case where a *stored* F is rounded by u_p after an
    accurate evaluation, which perturbs the stopping test itself slightly.
    """
    L = _grid(case.k)
    n = L.shape[0]
    u = P("u", np.zeros(n))
    nrm0 = _true_residual(L, u, case.lam)

    F = P("F", _residual(P, L, u, case.lam, case.fine_residual))
    stopped = False
    it = 0
    for it in range(1, case.maxit + 1):
        J = P("J", _jacobian(L, u, case.lam))
        try:
            delta = P("delta", np.linalg.solve(J, -F))
        except np.linalg.LinAlgError:
            break
        u = P("u", u + delta)
        F = P("F", _residual(P, L, u, case.lam, case.fine_residual))
        nrm = float(np.linalg.norm(F))
        if not np.isfinite(nrm):
            break
        if nrm <= case.rtol * nrm0:
            stopped = True
            break

    true_resid = _true_residual(L, u, case.lam) / nrm0
    converged = stopped and np.isfinite(true_resid) and true_resid <= 2.0 * case.rtol
    return {"converged": converged, "iterations": it, "true_resid": true_resid,
            "false_stop": stopped and not converged}


def kernel(P, case: Case | None = None):
    return newton(P, case or CASES[0])


def condition_number(case: Case) -> float:
    """kappa_2 of the Jacobian at the converged fp64 solution."""
    L = _grid(case.k)
    u = np.zeros(L.shape[0])
    for _ in range(case.maxit):
        F = L @ u - case.lam * np.exp(u)
        if np.linalg.norm(F) <= 1e-13 * max(1.0, np.linalg.norm(L @ u)):
            break
        u += np.linalg.solve(_jacobian(L, u, case.lam), -F)
    return float(np.linalg.cond(_jacobian(L, u, case.lam)))


#: Three conditionings, spanning about a decade in kappa, so the knee's
#: dependence on log2(kappa) can be checked rather than assumed.
CASES = (Case(k=16, lam=1.0), Case(k=16, lam=6.0), Case(k=24, lam=6.0))
