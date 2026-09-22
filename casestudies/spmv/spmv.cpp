// Tier-2 of the cost model (casestudies/newton/DESIGN.md): cost per iteration as
// a function of storage format, measured rather than modeled.
//
// A block CSR sparse matrix-vector product templated on the storage type of the
// matrix values, the storage type of the input vector, and the block size,
// accumulating and writing in double.  Block size 1 is scalar CSR, so AIJ and
// BAIJ share one code path and the comparison between them is exact.  This is
// the shape a mixed-precision solver wants: reduced storage to cut memory
// traffic, full precision in the accumulator so the operator's action is not
// degraded beyond what the storage already costs.
//
// Why the block size is the interesting axis.  Per block row, one SpMV moves
//
//     rb * (bs^2 * w + 4) + 4 + bs * (v + 8)   bytes,
//
// for rb blocks per block row, value width w and vector width v.  Narrowing w
// leaves the column indices, the row pointer and the vector traffic alone, so
// the saving is bounded by (8 + 4/bs^2) / (w + 4/bs^2).  In scalar CSR that is
// 1.5x for fp32 and 2x for fp16 -- the 4-byte column index per nonzero caps it.
// At bs = 3, the index costs 4/9 of a byte per scalar nonzero and the bound
// rises to 1.90x and 3.45x.  A blocked format is most of what makes reduced
// precision pay, which is why PETSc's BAIJ and examples like snes/tutorials/ex56
// (3D elasticity, bs = 3) are the right target.
//
// The `predict` column is this model; `speedup` is what was measured.
// test_stencil.cpp asserts the derivation on the host, with no GPU.

#include <Kokkos_Core.hpp>

#include "stencil.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

using Device = Kokkos::DefaultExecutionSpace;

using half_t = Kokkos::Experimental::half_t;
using bhalf_t = Kokkos::Experimental::bhalf_t;

template <class T> struct TypeName;
template <> struct TypeName<double> { static constexpr const char *value = "fp64"; };
template <> struct TypeName<float> { static constexpr const char *value = "fp32"; };
template <> struct TypeName<half_t> { static constexpr const char *value = "fp16"; };
template <> struct TypeName<bhalf_t> { static constexpr const char *value = "bf16"; };

// sizeof() is the honest width: Kokkos' half types fall back to float where the
// backend has no native support, and the bandwidth model must follow suit rather
// than claim a 2-byte load that is not happening.
template <class T> constexpr std::size_t stored_bytes() { return sizeof(T); }

// ------------------------------------------------------- block accumulator

template <int BS> struct BlockAcc {
  double v[BS];
  KOKKOS_INLINE_FUNCTION BlockAcc() {
    for (int i = 0; i < BS; ++i) v[i] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION BlockAcc &operator+=(const BlockAcc &o) {
    for (int i = 0; i < BS; ++i) v[i] += o.v[i];
    return *this;
  }
};

namespace Kokkos {
template <int BS> struct reduction_identity<BlockAcc<BS>> {
  KOKKOS_FORCEINLINE_FUNCTION static BlockAcc<BS> sum() { return BlockAcc<BS>(); }
};
}  // namespace Kokkos

// ------------------------------------------------------------------- kernel

template <class MatT, class VecT, int BS> struct BlockSpMV {
  using team_policy = Kokkos::TeamPolicy<Device>;
  using member_type = typename team_policy::member_type;

  Kokkos::View<const int *, Device> rowptr, colidx;
  Kokkos::View<const MatT *, Device> vals;
  Kokkos::View<const VecT *, Device> x;
  Kokkos::View<double *, Device> y;
  int nblock_rows, rows_per_team;

  KOKKOS_INLINE_FUNCTION void operator()(const member_type &team) const {
    const int first = team.league_rank() * rows_per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rows_per_team), [&](const int r) {
      const int brow = first + r;
      if (brow >= nblock_rows) return;
      const int begin = rowptr(brow), end = rowptr(brow + 1);

      BlockAcc<BS> acc;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, end - begin),
          [&](const int jj, BlockAcc<BS> &a) {
            const int b = begin + jj;
            const int col = colidx(b);
            const size_t base = static_cast<size_t>(b) * BS * BS;
            for (int i = 0; i < BS; ++i) {
              double s = 0.0;
              for (int j = 0; j < BS; ++j)
                s += static_cast<double>(vals(base + i * BS + j))
                   * static_cast<double>(x(col * BS + j));
              a.v[i] += s;
            }
          },
          Kokkos::Sum<BlockAcc<BS>>(acc));

      Kokkos::single(Kokkos::PerThread(team), [&]() {
        for (int i = 0; i < BS; ++i) y(brow * BS + i) = acc.v[i];
      });
    });
  }
};

struct Timing {
  double seconds = 1e30;
  double checksum = 0.0;
  int team_size = 0, vector_length = 0;
};

template <class MatT, class VecT, int BS>
static Timing time_spmv(const HostBCSR &A, int reps) {
  Kokkos::View<int *, Device> rowptr("rowptr", A.nblock_rows + 1);
  Kokkos::View<int *, Device> colidx("colidx", A.nblocks);
  Kokkos::View<MatT *, Device> vals("vals", A.nnz());
  Kokkos::View<VecT *, Device> x("x", A.rows());
  Kokkos::View<double *, Device> y("y", A.rows());

  {
    auto h_rowptr = Kokkos::create_mirror_view(rowptr);
    auto h_colidx = Kokkos::create_mirror_view(colidx);
    auto h_vals = Kokkos::create_mirror_view(vals);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i <= A.nblock_rows; ++i) h_rowptr(i) = A.rowptr[i];
    for (long long b = 0; b < A.nblocks; ++b) h_colidx(b) = A.colidx[b];
    for (long long e = 0; e < A.nnz(); ++e)
      h_vals(e) = static_cast<MatT>(static_cast<float>(A.vals[e]));
    for (int i = 0; i < A.rows(); ++i)
      h_x(i) = static_cast<VecT>(static_cast<float>(1.0 + 0.001 * (i % 97)));
    Kokkos::deep_copy(rowptr, h_rowptr);
    Kokkos::deep_copy(colidx, h_colidx);
    Kokkos::deep_copy(vals, h_vals);
    Kokkos::deep_copy(x, h_x);
  }

  const bool on_device =
      !std::is_same<Device, Kokkos::DefaultHostExecutionSpace>::value;
  const std::vector<int> vlens =
      on_device ? std::vector<int>{4, 8, 16, 32} : std::vector<int>{1};
  const std::vector<int> tsizes =
      on_device ? std::vector<int>{4, 8, 16} : std::vector<int>{1};

  Timing best;
  for (int vl : vlens)
    for (int ts : tsizes) {
      const int rows_per_team = ts;  // one block row per team thread
      const int leagues = (A.nblock_rows + rows_per_team - 1) / rows_per_team;
      BlockSpMV<MatT, VecT, BS> op{rowptr,          colidx, vals, x, y,
                                   A.nblock_rows,   rows_per_team};
      // Explicit team size and vector length: Kokkos' AUTO misbehaves here.
      typename BlockSpMV<MatT, VecT, BS>::team_policy policy(leagues, ts, vl);

      Kokkos::parallel_for("warmup", policy, op);
      Kokkos::fence();
      Kokkos::Timer timer;
      for (int r = 0; r < reps; ++r) Kokkos::parallel_for("spmv", policy, op);
      Kokkos::fence();
      const double s = timer.seconds() / reps;
      if (s < best.seconds) {
        // Reduce the result so the kernel cannot be optimised away, and so a
        // format that silently produced garbage is visible in the output.
        double sum = 0.0;
        Kokkos::parallel_reduce(
            "checksum", Kokkos::RangePolicy<Device>(0, A.rows()),
            KOKKOS_LAMBDA(const int i, double &a) { a += y(i); }, sum);
        best = Timing{s, sum, ts, vl};
      }
    }
  return best;
}

template <class MatT, class VecT, int BS>
static void report(const HostBCSR &A, int reps, double baseline_seconds) {
  const Timing t = time_spmv<MatT, VecT, BS>(A, reps);
  const double bytes = model_bytes(A, stored_bytes<MatT>(), stored_bytes<VecT>());
  const double base = model_bytes(A, sizeof(double), sizeof(double));
  std::printf("%-22s %-5s %-5s %9.3f %8.1f %8.2fx %8.2fx %5d %5d %13.5e\n",
              A.name.c_str(), TypeName<MatT>::value, TypeName<VecT>::value,
              t.seconds * 1e3, bytes / t.seconds / 1e9, baseline_seconds / t.seconds,
              base / bytes, t.team_size, t.vector_length, t.checksum);
  std::fflush(stdout);
}

template <int BS> static void sweep(const HostBCSR &A, int reps) {
  std::printf("# %s: block rows %d, blocks %lld, blocks/row %.1f, nnz/row %.1f\n",
              A.name.c_str(), A.nblock_rows, A.nblocks, A.blocks_per_row(),
              double(A.nnz()) / A.rows());
  const double base = time_spmv<double, double, BS>(A, reps).seconds;
  report<double, double, BS>(A, reps, base);
  report<float, double, BS>(A, reps, base);
  report<float, float, BS>(A, reps, base);
  report<bhalf_t, double, BS>(A, reps, base);
  report<half_t, double, BS>(A, reps, base);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int reps = 20, k_scalar = 200, k_block = 80;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
      if (a == "--small") { k_scalar = 100; k_block = 40; }
    }

    std::printf("# SpMV storage-format sweep; accumulator and output always fp64\n");
    std::printf("# exec space: %s\n", Device::name());
    std::printf("# stored widths (bytes): fp64=%zu fp32=%zu bf16=%zu fp16=%zu\n",
                stored_bytes<double>(), stored_bytes<float>(),
                stored_bytes<bhalf_t>(), stored_bytes<half_t>());
    std::printf("# 'predict' is the minimal-traffic byte ratio vs fp64/fp64 in the\n"
                "# same layout; index traffic is why it is below 2x, and why the\n"
                "# bs=3 rows beat the scalar ones.\n");
    std::printf("%-22s %-5s %-5s %9s %8s %9s %9s %5s %5s %13s\n", "matrix", "mat",
                "vec", "ms", "GB/s", "speedup", "predict", "team", "vlen", "checksum");

    // Scalar baselines.
    sweep<1>(stencil3d(k_scalar, 7, 1), reps);
    sweep<1>(stencil3d(96, 27, 1), reps);

    // ex56-shaped: 3D elasticity on trilinear hexahedra, 3 unknowns per node.
    // The same matrix is timed blocked and expanded to scalar, so the AIJ/BAIJ
    // difference is layout alone.
    const HostBCSR E = stencil3d(k_block, 27, 3);
    sweep<3>(E, reps);
    sweep<1>(expand_to_scalar(E), reps);
  }
  Kokkos::finalize();
  return 0;
}
