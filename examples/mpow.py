"""The worked example from Blame Analysis (ICSE'16), Figure 2.

Computes ``res = a^4 - 4a^3 + 6a^2 - 4a + 1`` -- that is, ``(a-1)^4`` -- at
a = 1.84089642, via repeated multiplication.  Intermediate terms reach ~25 while
the result is ~0.5, so the summation loses roughly 1.7 decimal digits to
cancellation.

The paper reports res = 0.5000000113 in double and 0.4999990927 with every
variable in single, and concludes that only `factor` inside mpow() can be
lowered -- it holds small exact integers.  Both are checked in tests/.
"""

from __future__ import annotations

A = 1.84089642

#: Several inputs, because one is not enough.  At a = 1.84089642 the result is
#: within 2.3e-8 of exactly 0.5, which an 8-bit format represents exactly -- so a
#: search given only this input can meet a loose tolerance by coarsening `res`
#: until it quantizes onto that lattice point, with a downstream format absorbing
#: upstream error rather than the computation being accurate.  These inputs
#: spread the result over three orders of magnitude, so no single grid point is
#: lucky for all of them.
CASES = (1.84089642, 1.2, 2.5, 0.6, 3.7)


def _mpow(P, a, factor, n):
    """res = factor * a**n, accumulated one multiply at a time."""
    res = P("mpow.res", factor)
    for _ in range(n):
        res = P("mpow.res", res * a)
    return res


def kernel(P, a_value=A):
    a = P("a", a_value)
    t1 = P("t1", 4.0 * a)
    t2 = P("t2", _mpow(P, a, P("mpow.factor", 6.0), 2))
    t3 = P("t3", _mpow(P, a, P("mpow.factor", 4.0), 3))
    t4 = P("t4", _mpow(P, a, P("mpow.factor", 1.0), 4))

    r1 = P("r1", t4 - t3)
    r2 = P("r2", r1 + t2)
    r3 = P("r3", r2 - t1)
    res = P("res", r3 + 1.0)
    return {"res": res}
