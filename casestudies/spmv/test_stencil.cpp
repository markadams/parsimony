// Host-only checks on the matrix and the bandwidth model, so a GPU job is never
// spent on a wrong matrix.  Build and run with:  make -f Makefile.test
//
//   c++ -std=c++17 -O2 test_stencil.cpp -o test_stencil && ./test_stencil

#include "stencil.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>

static int failures = 0;

#define CHECK(cond, ...)                                   \
  do {                                                     \
    if (!(cond)) {                                         \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
      std::printf(__VA_ARGS__);                            \
      std::printf("\n");                                   \
      ++failures;                                          \
    }                                                      \
  } while (0)

static void check_structure(int k, int points) {
  const HostCSR A = stencil3d(k, points);
  CHECK(A.n == k * k * k, "n = %d, expected %d", A.n, k * k * k);
  CHECK(static_cast<int>(A.rowptr.size()) == A.n + 1, "rowptr size");
  CHECK(A.rowptr.back() == static_cast<int>(A.nnz), "rowptr terminator");
  CHECK(A.vals.size() == A.colidx.size(), "vals/colidx length mismatch");

  // Interior rows have the full stencil; no row exceeds it.
  int max_row = 0;
  long long interior_rows = 0;
  for (int i = 0; i < A.n; ++i) {
    const int len = A.rowptr[i + 1] - A.rowptr[i];
    if (len > max_row) max_row = len;
    if (len == points) ++interior_rows;
    CHECK(len >= 1 && len <= points, "row %d has %d entries", i, len);
  }
  CHECK(max_row == points, "max row length %d, expected %d", max_row, points);
  CHECK(interior_rows == static_cast<long long>(k - 2) * (k - 2) * (k - 2),
        "interior rows %lld, expected %lld", interior_rows,
        static_cast<long long>(k - 2) * (k - 2) * (k - 2));

  // Column indices in range, and no duplicates within a row.
  for (int i = 0; i < A.n; ++i) {
    std::map<int, int> seen;
    for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
      CHECK(A.colidx[j] >= 0 && A.colidx[j] < A.n, "col %d out of range",
            A.colidx[j]);
      CHECK(++seen[A.colidx[j]] == 1, "duplicate column %d in row %d", A.colidx[j], i);
    }
  }

  // Symmetry: A(i,j) == A(j,i).
  std::map<long long, double> entry;
  for (int i = 0; i < A.n; ++i)
    for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j)
      entry[static_cast<long long>(i) * A.n + A.colidx[j]] = A.vals[j];
  int asym = 0;
  for (const auto &kv : entry) {
    const long long i = kv.first / A.n, j = kv.first % A.n;
    const auto it = entry.find(j * A.n + i);
    if (it == entry.end() || std::fabs(it->second - kv.second) > 1e-12) ++asym;
  }
  CHECK(asym == 0, "%d asymmetric entries", asym);

  // Strict diagonal dominance implies SPD here, and guarantees the CG-style
  // solves this matrix is meant to stand in for actually converge.
  int weak = 0;
  for (int i = 0; i < A.n; ++i) {
    double diag = 0.0, off = 0.0;
    for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
      if (A.colidx[j] == i) diag = A.vals[j];
      else off += std::fabs(A.vals[j]);
    }
    if (!(diag > off)) ++weak;
  }
  CHECK(weak == 0, "%d rows not strictly diagonally dominant", weak);

  // Entries must not be exactly representable in a low-precision format, or
  // storage would look free.  Approximate fp8e4m3 (3 significand bits).
  int exact8 = 0;
  for (double v : A.vals) {
    const double scaled = std::ldexp(v, 3 - static_cast<int>(std::ilogb(v)) - 1);
    if (scaled == std::round(scaled)) ++exact8;
  }
  CHECK(exact8 == 0, "%d of %zu entries exactly representable in 4 significand bits",
        exact8, A.vals.size());

  std::printf("ok  %-14s n=%7d nnz=%9lld nnz/row=%5.2f\n", A.name.c_str(), A.n, A.nnz,
              double(A.nnz) / A.n);
}

// Per row, one SpMV moves  r*(w + 4) + 4 + v + 8  bytes, for r nonzeros per row,
// value width w, vector width v.  Narrowing w leaves three terms untouched: the
// column indices, the row pointer, and the vector read and write.  So the saving
// is bounded by
//
//     ratio(w) = (r*12 + 20) / (r*w + 4r + 20)  ->  12/(w+4)  as r -> infinity,
//
// which is 1.5 for fp32 and 2 for fp16 only in the limit of infinitely fat rows.
// At a 7-point stencil (r ~ 6.5) it is 1.36 and 1.67.  This is why "halving the
// storage nearly halves the time" is wrong for CSR, and the test asserts the
// derivation rather than a remembered constant.
static void check_bandwidth_model() {
  auto expected = [](const HostCSR &A, double w, double v) {
    const double r = double(A.nnz) / A.n;
    return (r * (8.0 + 4.0) + 4.0 + 8.0 + 8.0) / (r * (w + 4.0) + 4.0 + v + 8.0);
  };

  for (int points : {7, 27}) {
    const HostCSR A = stencil3d(20, points);
    const double b64 = model_bytes(A, 8, 8);
    for (double w : {4.0, 2.0}) {
      const double got = b64 / model_bytes(A, size_t(w), 8);
      // Relative: the derivation charges n row pointers, model_bytes charges
      // n+1, an O(1/n) difference that is real and not worth papering over.
      CHECK(std::fabs(got - expected(A, w, 8)) / got < 1e-5,
            "%s: %g-byte values give ratio %.6f, derivation says %.6f", A.name.c_str(),
            w, got, expected(A, w, 8));
      CHECK(got < 12.0 / (w + 4.0) + 1e-9,
            "%s: ratio %.4f exceeds the infinite-row bound %.4f", A.name.c_str(), got,
            12.0 / (w + 4.0));
    }
    std::printf("ok  %-10s fp32 saves %.2fx, fp16 saves %.2fx"
                "  (bounds: %.2fx, %.2fx at infinite nnz/row)\n",
                A.name.c_str(), b64 / model_bytes(A, 4, 8), b64 / model_bytes(A, 2, 8),
                12.0 / 8.0, 12.0 / 6.0);
  }

  // Fatter rows dilute the index and vector traffic, so the saving grows.
  const HostCSR thin = stencil3d(20, 7), fat = stencil3d(20, 27);
  CHECK(model_bytes(fat, 8, 8) / model_bytes(fat, 4, 8)
            > model_bytes(thin, 8, 8) / model_bytes(thin, 4, 8),
        "27-point should benefit more from narrow values than 7-point");

  // Narrowing the *vector* too buys almost nothing: it is O(n), not O(nnz).
  const double v_only = model_bytes(thin, 8, 8) / model_bytes(thin, 8, 2);
  CHECK(v_only < 1.07, "narrowing only the vector gave %.3f, expected < 1.07", v_only);
  std::printf("ok  narrowing only the vector: %.3fx  (O(n) traffic, not O(nnz))\n",
              v_only);
}

int main() {
  check_structure(8, 7);
  check_structure(8, 27);
  check_structure(12, 7);
  check_bandwidth_model();
  std::printf(failures ? "\n%d FAILURES\n" : "\nall checks passed\n", failures);
  return failures ? 1 : 0;
}
