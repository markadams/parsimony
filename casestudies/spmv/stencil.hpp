// Matrix generation and the bandwidth model, deliberately free of Kokkos so they
// can be compiled and checked on a laptop (see test_stencil.cpp).  The GPU sweep
// in spmv.cpp includes this.
//
// Everything here is block CSR with block size bs; bs = 1 is scalar CSR, so the
// two live on one code path and the comparison between them is exact rather than
// approximate.

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

// Block compressed sparse row, in PETSc's BAIJ layout: one column index per
// block, values stored bs*bs contiguously per block (row-major within a block).
struct HostBCSR {
  std::vector<int> rowptr, colidx;  // indexed by block row / block
  std::vector<double> vals;         // nblocks * bs * bs
  int nblock_rows = 0;
  int bs = 1;
  long long nblocks = 0;
  std::string name;

  int rows() const { return nblock_rows * bs; }
  long long nnz() const { return nblocks * static_cast<long long>(bs) * bs; }
  double blocks_per_row() const { return double(nblocks) / nblock_rows; }
};

// A structured stencil on a k^3 grid of nodes with `bs` unknowns per node:
// 7-point (scalar diffusion) or 27-point (trilinear hexahedra, which is what
// PETSc's snes/tutorials/ex56 elasticity assembles, with bs = 3).
//
// Entries are generic rather than small exact integers, for the reason the
// conjugate-gradient study documents: a constant-coefficient stencil has entries
// exactly representable in every format down to 8 bits, which makes matrix
// storage look free when it is not.  Diagonal blocks are symmetric positive
// definite and dominate their row, so the operator stands in for a real one.
inline HostBCSR stencil3d(int k, int points, int bs = 1) {
  HostBCSR A;
  A.bs = bs;
  A.nblock_rows = k * k * k;
  A.name = std::to_string(points) + "pt-" + std::to_string(k) + "^3-bs"
         + std::to_string(bs);
  A.rowptr.reserve(A.nblock_rows + 1);
  A.rowptr.push_back(0);

  // Coupling between unknowns i and j at the same node: symmetric in (i,j) and
  // non-dyadic.  This is what makes the diagonal block full, as an elasticity
  // block is, rather than a scaled identity.
  auto intra = [](int node, int i, int j) {
    if (i == j) return 0.0;
    return 0.3 * std::sin(1.7 * node + 2.3 * (i + j) + 0.4 * std::abs(i - j))
         / (1.0 + std::abs(i - j));
  };

  // Coupling between node (x,y,z) and neighbour (xx,yy,zz), unknowns i and j.
  // Symmetric in the two endpoints and in (i,j), so the whole matrix is
  // symmetric; non-dyadic, so nothing is free to store.
  auto coupling = [](int x, int y, int z, int xx, int yy, int zz, int i, int j) {
    return (1.0 + 0.25 * std::sin(0.7 * (x + xx) + 1.3 * (y + yy) + 2.1 * (z + zz)
                                  + 0.9 * (i + j)))
         / (1.0 + std::abs(i - j));
  };

  auto in_stencil = [points](int dx, int dy, int dz) {
    return !(points == 7 && (std::abs(dx) + std::abs(dy) + std::abs(dz)) > 1);
  };

  std::vector<double> offrow(bs, 0.0);  // absolute row sum from off-diagonal blocks

  for (int z = 0; z < k; ++z)
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x) {
        const int node = (z * k + y) * k + x;

        // First pass: how much absolute weight the off-diagonal blocks put in
        // each scalar row, which sets how large the diagonal has to be.
        std::fill(offrow.begin(), offrow.end(), 0.0);
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              if (!in_stencil(dx, dy, dz)) continue;
              const int xx = x + dx, yy = y + dy, zz = z + dz;
              if (xx < 0 || xx >= k || yy < 0 || yy >= k || zz < 0 || zz >= k) continue;
              for (int i = 0; i < bs; ++i)
                for (int j = 0; j < bs; ++j)
                  offrow[i] += coupling(x, y, z, xx, yy, zz, i, j);
            }

        // Second pass in the same order, emitting the diagonal block in its
        // place rather than appending it.  The (dz,dy,dx) loop visits nodes in
        // increasing linear index, so column indices come out sorted, as PETSc
        // requires and as gives the input vector its natural access order.
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (!in_stencil(dx, dy, dz)) continue;
              const int xx = x + dx, yy = y + dy, zz = z + dz;
              if (xx < 0 || xx >= k || yy < 0 || yy >= k || zz < 0 || zz >= k) continue;

              if (dx == 0 && dy == 0 && dz == 0) {
                // Diagonal block: full and symmetric, with a diagonal large
                // enough that every scalar row stays strictly dominant.
                A.colidx.push_back(node);
                for (int i = 0; i < bs; ++i) {
                  double intra_row = 0.0;
                  for (int j = 0; j < bs; ++j)
                    intra_row += std::fabs(intra(node, i, j));
                  for (int j = 0; j < bs; ++j)
                    A.vals.push_back(i == j ? offrow[i] + intra_row + 1.0
                                            : intra(node, i, j));
                }
              } else {
                A.colidx.push_back((zz * k + yy) * k + xx);
                for (int i = 0; i < bs; ++i)
                  for (int j = 0; j < bs; ++j)
                    A.vals.push_back(-coupling(x, y, z, xx, yy, zz, i, j));
              }
            }
        A.rowptr.push_back(static_cast<int>(A.colidx.size()));
      }
  A.nblocks = static_cast<long long>(A.colidx.size());
  return A;
}

// Minimal memory traffic for one SpMV in BAIJ layout: each block costs bs*bs
// values plus ONE 4-byte column index, each block row a row pointer, the input
// vector is read once and the output written once.
inline double model_bytes(const HostBCSR &A, std::size_t mat_bytes,
                          std::size_t vec_bytes) {
  const double bs2 = double(A.bs) * A.bs;
  return double(A.nblocks) * (bs2 * mat_bytes + sizeof(int))
       + double(A.nblock_rows + 1) * sizeof(int)
       + double(A.rows()) * (vec_bytes + sizeof(double));
}

// The same matrix stored as scalar AIJ: identical values, but one column index
// per *scalar* nonzero and one row pointer per scalar row.
//
// The difference between this and model_bytes() is the entire BAIJ advantage,
// and it is all index traffic: 4 bytes per scalar nonzero against 4/bs^2.
inline double model_bytes_as_aij(const HostBCSR &A, std::size_t mat_bytes,
                                 std::size_t vec_bytes) {
  return double(A.nnz()) * (mat_bytes + sizeof(int))
       + double(A.rows() + 1) * sizeof(int)
       + double(A.rows()) * (vec_bytes + sizeof(double));
}

// Expand to bs = 1, so the same matrix can be timed in both layouts.  Comparing
// a blocked matrix against a separately generated scalar one would confound the
// layout with the problem; this does not.
inline HostBCSR expand_to_scalar(const HostBCSR &A) {
  if (A.bs == 1) return A;
  const int bs = A.bs;
  HostBCSR S;
  S.bs = 1;
  S.nblock_rows = A.rows();
  S.name = A.name + "-as-aij";
  S.rowptr.reserve(S.nblock_rows + 1);
  S.rowptr.push_back(0);
  for (int bi = 0; bi < A.nblock_rows; ++bi)
    for (int i = 0; i < bs; ++i) {
      for (int b = A.rowptr[bi]; b < A.rowptr[bi + 1]; ++b)
        for (int j = 0; j < bs; ++j) {
          S.colidx.push_back(A.colidx[b] * bs + j);
          S.vals.push_back(A.vals[static_cast<size_t>(b) * bs * bs + i * bs + j]);
        }
      S.rowptr.push_back(static_cast<int>(S.colidx.size()));
    }
  S.nblocks = static_cast<long long>(S.colidx.size());
  return S;
}
