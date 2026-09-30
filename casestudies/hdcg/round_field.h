#ifndef PARSIMONY_ROUND_FIELD_H
#define PARSIMONY_ROUND_FIELD_H
/* Apply storage rounding to Grid lattice fields and to raw BLAS buffers.
 *
 * What this does: takes each double in the field's storage and snaps it to the
 * target format, in place, on the device.  Arithmetic afterwards is unchanged
 * fp64 -- the study measures how few bits the *stored* input needs, not
 * reduced-precision arithmetic.  Rounding every double in a `vobj` rounds the
 * real and imaginary parts of every SIMD lane, which is what "per real
 * component" means for a ComplexD field.
 *
 *   #include <Grid/Grid.h>
 *   #include "round_field.h"
 *   Parsimony::RoundField(Aggregates.subspace[b], spec);
 *
 * Note that `autoView` and `accelerator_for` are macros, so they appear here
 * unqualified even though everything around them is namespace-qualified.
 */

#include <vector>

#include "round_bits.h"

namespace Parsimony {

/* Only double-precision fields may be rounded.  A ComplexF field would pass a
 * naive sizeof() check (8 bytes, divisible by sizeof(double)) and then be
 * reinterpreted as garbage, so the scalar type is matched by name instead.
 */
template <class T> struct IsDoubleScalar { static const bool value = false; };
template <> struct IsDoubleScalar<Grid::ComplexD> { static const bool value = true; };
template <> struct IsDoubleScalar<Grid::RealD> { static const bool value = true; };

/* Round a lattice field in place; returns the relative L2 change it caused.
 *
 * The return value is the positive check that rounding happened: a sweep whose
 * knee sits at 8 bits but whose logged change is 0.0 everywhere is measuring a
 * no-op, which is exactly the failure a missing hook produces.
 */
template <class vobj>
double RoundField(Grid::Lattice<vobj> &field, const RoundSpec &spec) {
  // autoView and accelerator_for expand to unqualified Grid names
  // (ViewCloser, acceleratorThreads, computeStream, ...), which will not
  // resolve from inside namespace Parsimony without this.
  using namespace Grid;

  typedef typename vobj::scalar_type scalar_type;
  static_assert(IsDoubleScalar<scalar_type>::value,
                "RoundField: double-precision fields only");
  static_assert(sizeof(vobj) % sizeof(double) == 0, "RoundField: unexpected padding");

  if (spec.identity()) return 0.0;

  Lattice<vobj> before(field.Grid());
  before = field;
  const double norm_before = norm2(field);

  const int words = (int)(sizeof(vobj) / sizeof(double));
  const uint64_t nsite = field.Grid()->oSites();
  {
    autoView(view, field, AcceleratorWrite);
    accelerator_for(ss, nsite, 1, {
      double *p = (double *)&view[ss];
      for (int w = 0; w < words; w++) p[w] = round_bits(p[w], spec);
    });
  }

  Lattice<vobj> diff(field.Grid());
  diff = before - field;
  const double norm_delta = norm2(diff);
  return norm_before > 0.0 ? std::sqrt(norm_delta / norm_before) : 0.0;
}

/* Round a whole subspace / eigenbasis.  Returns the largest relative change
 * over the vectors, for the same positive check.
 */
template <class Field>
double RoundFields(std::vector<Field> &fields, const RoundSpec &spec) {
  double worst = 0.0;
  for (size_t i = 0; i < fields.size(); i++) {
    const double d = RoundField(fields[i], spec);
    if (d > worst) worst = d;
  }
  return worst;
}

/* Round a host-resident buffer of complex doubles.  `n` counts complex
 * entries, not doubles.
 */
inline double RoundComplexBuffer(Grid::ComplexD *data, uint64_t n,
                                 const RoundSpec &spec) {
  if (spec.identity()) return 0.0;
  double num = 0.0, den = 0.0;
  double *p = (double *)data;
  for (uint64_t w = 0; w < 2 * n; w++) {
    const double a = p[w];
    const double b = round_bits(a, spec);
    den += a * a;
    num += (a - b) * (a - b);
    p[w] = b;
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

}  // namespace Parsimony

#endif /* PARSIMONY_ROUND_FIELD_H */
