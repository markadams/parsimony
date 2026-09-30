#ifndef PARSIMONY_ROUND_BITS_H
#define PARSIMONY_ROUND_BITS_H
/* Storage rounding of a double to a virtual or named floating-point format.
 *
 * This is the C++ twin of parsimony's `formats.round_to`, and it must agree
 * with it exactly: the Python side chooses configurations and the C++ side
 * applies them, so any disagreement silently mislabels a knee.  The agreement
 * is checked on a few hundred doubles by tests/test_hdcg_round.py, which
 * compiles round_bits_probe.cpp and diffs against round_to.
 *
 * Deliberately free of Grid (and of any device intrinsics) so that test exists:
 * the Grid `Lattice` wrapper lives in round_field.h.  Everything here is
 * constexpr-friendly, header-only, and callable from a CUDA device function.
 */

#include <cmath>
#include <cstring>
#include <cstdlib>

#ifdef __CUDACC__
#define PARSIMONY_HD __host__ __device__
#else
#define PARSIMONY_HD
#endif

namespace Parsimony {

/* A binary format, named the way IEEE 754 names them: `mant_bits` counts the
 * *explicit* significand bits, `precision` adds the implicit leading one.
 * Virtual formats (`v40`) carry float64's exponent range and k significand
 * bits; they answer "how many bits does this value need?" without committing
 * to a storage layout.
 */
struct RoundSpec {
  int precision;  /* significand bits including the implicit leading bit */
  int exp_bits;

  PARSIMONY_HD int mant_bits() const { return precision - 1; }
  PARSIMONY_HD int emax() const { return (1 << (exp_bits - 1)) - 1; }
  PARSIMONY_HD int emin() const { return 1 - emax(); }
  PARSIMONY_HD double max_finite() const {
    return (2.0 - ldexp(1.0, -mant_bits())) * ldexp(1.0, emax());
  }
  /* float64 and anything wider is a no-op: this is the "rounding off" rung. */
  PARSIMONY_HD bool identity() const { return precision >= 53 && exp_bits >= 11; }
};

static const RoundSpec ROUND_OFF = {53, 11};

/* Round to nearest representable value, ties to even.
 *
 * Mirrors round_to() branch for branch, including the order in which overflow,
 * zero and non-finite inputs are resolved.  nearbyint() under the default
 * FE_TONEAREST is ties-to-even, which is what np.round does; the products and
 * quotients below are all exact (scale is a power of two and the significand
 * fits), so the two implementations agree bit for bit rather than nearly.
 */
PARSIMONY_HD inline double round_bits(double a, const RoundSpec &s) {
  if (s.identity()) return a;
  if (a == 0.0) return a;
#ifdef __CUDA_ARCH__
  if (!isfinite(a)) return a;
#else
  if (!std::isfinite(a)) return a;
#endif

  int e;
  const double m = frexp(a, &e); /* a == m * 2^e, 0.5 <= |m| < 1 */
  const int unbiased = e - 1;    /* the IEEE exponent E in a == +-1.f * 2^E */

  double y;
  if (unbiased < s.emin()) {
    /* Subnormal in the target format: quantize to the smallest step.  Only
     * reachable for formats whose exponent range is narrower than the value's,
     * which is why round_to() guards this branch -- for a virtual format the
     * quantum itself underflows.
     */
    const double quantum = ldexp(1.0, s.emin() - s.mant_bits());
    y = nearbyint(a / quantum) * quantum;
  } else {
    const double scale = ldexp(1.0, s.precision);
    y = ldexp(nearbyint(m * scale) / scale, e);
  }

  if (fabs(y) > s.max_finite()) y = copysign(INFINITY, a);
  return y;
}

/* Parse "v40", "fp64", "bf16", ... into a spec.  Returns false on anything
 * unrecognized; callers should treat that as a usage error rather than
 * defaulting, so a typo in a sweep script cannot quietly run at fp64.
 */
inline bool parse_round_spec(const char *text, RoundSpec &out) {
  if (text == 0 || text[0] == '\0') return false;

  struct Named { const char *name; int precision; int exp_bits; };
  static const Named table[] = {
      {"fp64", 53, 11}, {"fp32", 24, 8},  {"tf32", 11, 8},
      {"bf16", 8, 8},   {"fp16", 11, 5},  {"fp8e5m2", 3, 5},
      {"fp8e4m3", 4, 4},
  };
  for (unsigned i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    if (std::strcmp(text, table[i].name) == 0) {
      out.precision = table[i].precision;
      out.exp_bits = table[i].exp_bits;
      return true;
    }
  }

  if (text[0] == 'v') {
    char *end = 0;
    const long bits = std::strtol(text + 1, &end, 10);
    if (end == text + 1 || *end != '\0' || bits < 1 || bits > 53) return false;
    out.precision = (int)bits;
    out.exp_bits = 11;
    return true;
  }

  /* A bare integer is accepted as shorthand for the virtual format, so
   * `--round-stencil 24` and `--round-stencil v24` mean the same thing. */
  {
    char *end = 0;
    const long bits = std::strtol(text, &end, 10);
    if (end != text && *end == '\0' && bits >= 1 && bits <= 53) {
      out.precision = (int)bits;
      out.exp_bits = 11;
      return true;
    }
  }
  return false;
}

}  // namespace Parsimony

#endif  /* PARSIMONY_ROUND_BITS_H */
