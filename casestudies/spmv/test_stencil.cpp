// Host-only checks on the matrix and the bandwidth model, so a GPU job is never
// spent on a wrong matrix or a wrong claim.
//
//   c++ -std=c++17 -O2 test_stencil.cpp -o test_stencil && ./test_stencil

#include "stencil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

static int failures = 0;

#define CHECK(cond, ...)                               \
  do {                                                 \
    if (!(cond)) {                                     \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                        \
      std::printf("\n");                               \
      ++failures;                                      \
    }                                                  \
  } while (0)

// Expand a block matrix to dense, for structural checks that are clearest there.
static std::vector<double> dense(const HostBCSR &A) {
  const int n = A.rows();
  std::vector<double> D(static_cast<size_t>(n) * n, 0.0);
  for (int bi = 0; bi < A.nblock_rows; ++bi)
    for (int b = A.rowptr[bi]; b < A.rowptr[bi + 1]; ++b) {
      const int bj = A.colidx[b];
      for (int i = 0; i < A.bs; ++i)
        for (int j = 0; j < A.bs; ++j)
          D[static_cast<size_t>(bi * A.bs + i) * n + (bj * A.bs + j)] =
              A.vals[static_cast<size_t>(b) * A.bs * A.bs + i * A.bs + j];
    }
  return D;
}

static void check_structure(int k, int points, int bs) {
  const HostBCSR A = stencil3d(k, points, bs);
  CHECK(A.nblock_rows == k * k * k, "block rows %d", A.nblock_rows);
  CHECK(static_cast<int>(A.rowptr.size()) == A.nblock_rows + 1, "rowptr size");
  CHECK(A.rowptr.back() == static_cast<int>(A.nblocks), "rowptr terminator");
  CHECK(A.vals.size() == static_cast<size_t>(A.nblocks) * bs * bs, "vals length");

  int max_row = 0;
  long long interior = 0;
  for (int i = 0; i < A.nblock_rows; ++i) {
    const int len = A.rowptr[i + 1] - A.rowptr[i];
    max_row = std::max(max_row, len);
    if (len == points) ++interior;
    CHECK(len >= 1 && len <= points, "block row %d has %d blocks", i, len);
  }
  CHECK(max_row == points, "max blocks/row %d, expected %d", max_row, points);
  CHECK(interior == static_cast<long long>(k - 2) * (k - 2) * (k - 2),
        "interior block rows %lld", interior);

  for (int i = 0; i < A.nblock_rows; ++i) {
    std::map<int, int> seen;
    for (int b = A.rowptr[i]; b < A.rowptr[i + 1]; ++b) {
      CHECK(A.colidx[b] >= 0 && A.colidx[b] < A.nblock_rows, "block col out of range");
      CHECK(++seen[A.colidx[b]] == 1, "duplicate block column in row %d", i);
    }
  }

  const int n = A.rows();
  const std::vector<double> D = dense(A);

  int asym = 0, weak = 0;
  for (int i = 0; i < n; ++i) {
    double off = 0.0;
    for (int j = 0; j < n; ++j) {
      if (std::fabs(D[size_t(i) * n + j] - D[size_t(j) * n + i]) > 1e-12) ++asym;
      if (i != j) off += std::fabs(D[size_t(i) * n + j]);
    }
    if (!(D[size_t(i) * n + i] > off)) ++weak;
  }
  CHECK(asym == 0, "%d asymmetric entries", asym);
  CHECK(weak == 0, "%d rows not strictly diagonally dominant", weak);

  // Entries must not be exactly representable in a low-precision format, or
  // storage would look free -- the Laplacian trap from the CG case study.
  int exact = 0;
  for (double v : A.vals) {
    if (v == 0.0) continue;
    const double scaled = std::ldexp(v, 3 - std::ilogb(v) - 1);
    if (scaled == std::round(scaled)) ++exact;
  }
  CHECK(exact == 0, "%d of %zu entries exact in 4 significand bits", exact,
        A.vals.size());

  std::printf("ok  %-18s rows=%7d blocks=%8lld blocks/row=%5.2f nnz/row=%6.2f\n",
              A.name.c_str(), n, A.nblocks, A.blocks_per_row(),
              double(A.nnz()) / n);
}

// Per block row, one SpMV moves  rb*(bs^2*w + 4) + 4 + bs*(v + 8)  bytes in BAIJ,
// against  rb*bs^2*(w + 4) + bs*4 + bs*(v + 8)  for the same matrix as scalar AIJ.
// The whole difference is index traffic: 4 bytes per scalar nonzero in AIJ, 4/bs^2
// in BAIJ.  Narrowing the value width therefore saves at most
//
//     (8 + 4/bs^2) / (w + 4/bs^2)
//
// in the fat-row limit: 1.50x (fp32) and 2.00x (fp16) at bs=1, but 1.90x and 3.45x
// at bs=3.  A blocked format is most of what makes reduced precision pay.
static void check_bandwidth_model() {
  std::printf("\n%-18s %-6s %8s %8s %8s %8s\n", "matrix", "format", "AIJ B/row",
              "BAIJ B/row", "AIJ gain", "BAIJ gain");
  for (int bs : {1, 3}) {
    const HostBCSR A = stencil3d(20, 27, bs);
    const double a64 = model_bytes_as_aij(A, 8, 8), b64 = model_bytes(A, 8, 8);
    for (size_t w : {size_t(4), size_t(2)}) {
      const double a = model_bytes_as_aij(A, w, 8), b = model_bytes(A, w, 8);
      std::printf("%-18s %-6s %8.0f %8.0f %7.2fx %7.2fx\n", A.name.c_str(),
                  w == 4 ? "fp32" : "fp16", a / A.nblock_rows, b / A.nblock_rows,
                  a64 / a, b64 / b);

      // Both are bounded by the fat-row limit for their own index overhead.
      const double idx_per_nnz = 4.0 / (bs * bs);
      CHECK(b64 / b < (8.0 + idx_per_nnz) / (double(w) + idx_per_nnz) + 1e-6,
            "bs=%d %zu-byte: BAIJ gain %.3f exceeds bound", bs, w, b64 / b);
      CHECK(a64 / a < 12.0 / (double(w) + 4.0) + 1e-6,
            "bs=%d %zu-byte: AIJ gain %.3f exceeds bound 1.5/2.0", bs, w, a64 / a);
      // Blocking never hurts, and at bs > 1 it strictly helps.
      CHECK(b64 / b >= a64 / a - 1e-9, "BAIJ gain below AIJ gain at bs=%d", bs);
      if (bs > 1) CHECK(b64 / b > a64 / a + 0.2, "bs=%d: BAIJ gain %.2f barely beats "
                        "AIJ %.2f", bs, b64 / b, a64 / a);
    }
    // At bs = 1 the two layouts are the same thing and must agree exactly.
    if (bs == 1)
      CHECK(std::fabs(a64 - b64) / b64 < 1e-9, "bs=1: AIJ and BAIJ models disagree");
  }

  // Blocking alone, before any precision change, and the combination.
  const HostBCSR E = stencil3d(20, 27, 3);
  const double aij64 = model_bytes_as_aij(E, 8, 8);
  std::printf("\nok  bs=3 blocking alone (fp64):        %.2fx less traffic than AIJ\n",
              aij64 / model_bytes(E, 8, 8));
  std::printf("ok  bs=3 BAIJ fp32 vs scalar AIJ fp64: %.2fx\n",
              aij64 / model_bytes(E, 4, 8));
  std::printf("ok  bs=3 BAIJ fp16 vs scalar AIJ fp64: %.2fx\n",
              aij64 / model_bytes(E, 2, 8));
  CHECK(aij64 / model_bytes(E, 4, 8) > 2.5, "bs=3 fp32 vs AIJ fp64 should exceed 2.5x");

  // Narrowing the vector alone is O(n) against the matrix's O(nnz): negligible.
  const HostBCSR T = stencil3d(20, 7, 1);
  const double v_only = model_bytes(T, 8, 8) / model_bytes(T, 8, 2);
  CHECK(v_only < 1.07, "narrowing only the vector gave %.3f", v_only);
  std::printf("ok  narrowing only the vector:         %.3fx (O(n), not O(nnz))\n",
              v_only);
}

// The AIJ/BAIJ comparison is only meaningful if both layouts hold the same
// matrix, so this checks the expansion element by element rather than trusting it.
static void check_expansion() {
  for (int bs : {2, 3}) {
    const HostBCSR B = stencil3d(6, 27, bs);
    const HostBCSR S = expand_to_scalar(B);
    CHECK(S.bs == 1, "expansion did not reach bs=1");
    CHECK(S.rows() == B.rows(), "row count changed: %d vs %d", S.rows(), B.rows());
    CHECK(S.nnz() == B.nnz(), "nonzero count changed: %lld vs %lld", S.nnz(), B.nnz());

    const std::vector<double> db = dense(B), ds = dense(S);
    CHECK(db.size() == ds.size(), "dense sizes differ");
    size_t differing = 0;
    for (size_t i = 0; i < db.size(); ++i)
      if (std::fabs(db[i] - ds[i]) > 0.0) ++differing;
    CHECK(differing == 0, "bs=%d: %zu entries differ after expansion", bs, differing);

    // Rows must be sorted by column within each scalar row, as CSR expects.
    int unsorted = 0;
    for (int i = 0; i < S.nblock_rows; ++i)
      for (int j = S.rowptr[i] + 1; j < S.rowptr[i + 1]; ++j)
        if (S.colidx[j] <= S.colidx[j - 1]) ++unsorted;
    CHECK(unsorted == 0, "bs=%d: %d out-of-order column indices", bs, unsorted);

    std::printf("ok  expansion bs=%d -> bs=1 preserves every entry (%lld nonzeros)\n",
                bs, S.nnz());
  }
}

int main() {
  check_structure(8, 7, 1);
  check_structure(8, 27, 1);
  check_structure(8, 27, 3);   // ex56-shaped: trilinear hexahedra, 3 dof/node
  check_structure(6, 7, 3);
  check_expansion();
  check_bandwidth_model();
  std::printf(failures ? "\n%d FAILURES\n" : "\nall checks passed\n", failures);
  return failures ? 1 : 0;
}
