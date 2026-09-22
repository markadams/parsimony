// Matrix generation and the bandwidth model, deliberately free of Kokkos so they
// can be compiled and checked on a laptop (see test_stencil.cpp).  The GPU sweep
// in spmv.cpp includes this.

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

struct HostCSR {
  std::vector<int> rowptr, colidx;
  std::vector<double> vals;
  int n = 0;
  long long nnz = 0;
  std::string name;
};

// A structured stencil on a k^3 grid: 7-point or 27-point, with Dirichlet
// boundaries folded into the diagonal so the operator stays symmetric positive
// definite and strictly diagonally dominant.
//
// Entries are generic rather than small exact integers, for the reason the
// conjugate-gradient study documents: a constant-coefficient stencil has entries
// that are exactly representable in every format down to 8 bits, which makes
// matrix storage look free when it is not.
inline HostCSR stencil3d(int k, int points) {
  HostCSR A;
  A.n = k * k * k;
  A.name = std::to_string(points) + "pt-" + std::to_string(k) + "^3";
  A.rowptr.reserve(A.n + 1);
  A.rowptr.push_back(0);

  for (int z = 0; z < k; ++z)
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x) {
        double offdiag_sum = 0.0;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              if (points == 7 && (std::abs(dx) + std::abs(dy) + std::abs(dz)) > 1)
                continue;
              const int xx = x + dx, yy = y + dy, zz = z + dz;
              if (xx < 0 || xx >= k || yy < 0 || yy >= k || zz < 0 || zz >= k) continue;
              // Symmetric in the two endpoints, so the matrix is symmetric.
              const double c =
                  2.0 + std::sin(0.7 * (x + xx) + 1.3 * (y + yy) + 2.1 * (z + zz));
              A.colidx.push_back((zz * k + yy) * k + xx);
              A.vals.push_back(-c);
              offdiag_sum += c;
            }
        A.colidx.push_back((z * k + y) * k + x);
        A.vals.push_back(offdiag_sum + 1.0);  // strictly diagonally dominant
        A.rowptr.push_back(static_cast<int>(A.colidx.size()));
      }
  A.nnz = static_cast<long long>(A.colidx.size());
  return A;
}

// Minimal memory traffic for one SpMV: each nonzero costs its value plus a
// 4-byte column index, each row a row pointer, the input vector is read once and
// the output written once.
//
// The index term is why reduced precision does not halve the time: at 7 points
// per row, fp64 moves 12 bytes per nonzero and fp32 moves 8.
inline double model_bytes(const HostCSR &A, std::size_t mat_bytes,
                          std::size_t vec_bytes) {
  return static_cast<double>(A.nnz) * (mat_bytes + sizeof(int))
       + static_cast<double>(A.n + 1) * sizeof(int)
       + static_cast<double>(A.n) * (vec_bytes + sizeof(double));
}
