// Tier-2 of the cost model (casestudies/newton/DESIGN.md): cost per iteration as
// a function of storage format, measured rather than modeled.
//
// A CSR sparse matrix-vector product templated on the storage type of the matrix
// values and of the input vector, accumulating and writing in double.  This is
// the shape a mixed-precision solver actually wants: reduced storage to cut
// memory traffic, full precision in the accumulator so the operator's action is
// not degraded beyond what the storage already costs.
//
// What the numbers mean.  Effective bandwidth uses the minimal-traffic model:
// each nonzero costs sizeof(MatT) for the value plus sizeof(int) for its column
// index; each row costs one row-pointer; the input vector is read once and the
// output written once.  Reported speedups are against fp64/fp64 on the same
// matrix and the same tuning sweep, so kernel quality largely cancels.
//
// The index traffic is the point, and it is worse than it first looks.  Per row,
// one SpMV moves r*(w+4) + 4 + v + 8 bytes for r nonzeros per row, value width w
// and vector width v.  Narrowing w leaves three terms alone: the column indices,
// the row pointer, and the vector read and write.  The saving is therefore
// bounded by 12/(w+4) -- 1.5x for fp32, 2x for fp16 -- and reaches that bound
// only as r goes to infinity.  At a 7-point stencil it is 1.36x and 1.67x.
// The `predict` column is this model; `speedup` is what was measured.
// See test_stencil.cpp, which asserts the derivation.

#include <Kokkos_Core.hpp>

#include "stencil.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
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
template <class T> constexpr size_t stored_bytes() { return sizeof(T); }

// ------------------------------------------------------------------- kernel

template <class MatT, class VecT> struct SpMV {
  using team_policy = Kokkos::TeamPolicy<Device>;
  using member_type = typename team_policy::member_type;

  Kokkos::View<const int *, Device> rowptr, colidx;
  Kokkos::View<const MatT *, Device> vals;
  Kokkos::View<const VecT *, Device> x;
  Kokkos::View<double *, Device> y;
  int n, rows_per_team;

  KOKKOS_INLINE_FUNCTION void operator()(const member_type &team) const {
    const int first = team.league_rank() * rows_per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, rows_per_team), [&](const int r) {
      const int row = first + r;
      if (row >= n) return;
      const int begin = rowptr(row), end = rowptr(row + 1);
      double sum = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, end - begin),
          [&](const int jj, double &acc) {
            const int j = begin + jj;
            acc += static_cast<double>(vals(j)) * static_cast<double>(x(colidx(j)));
          },
          sum);
      Kokkos::single(Kokkos::PerThread(team), [&]() { y(row) = sum; });
    });
  }
};

struct Timing {
  double seconds = 1e30;
  double checksum = 0.0;
  int team_size = 0, vector_length = 0;
};

template <class MatT, class VecT>
static Timing time_spmv(const HostCSR &A, int reps) {
  Kokkos::View<int *, Device> rowptr("rowptr", A.n + 1), colidx("colidx", A.nnz);
  Kokkos::View<MatT *, Device> vals("vals", A.nnz);
  Kokkos::View<VecT *, Device> x("x", A.n);
  Kokkos::View<double *, Device> y("y", A.n);

  {
    auto h_rowptr = Kokkos::create_mirror_view(rowptr);
    auto h_colidx = Kokkos::create_mirror_view(colidx);
    auto h_vals = Kokkos::create_mirror_view(vals);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i <= A.n; ++i) h_rowptr(i) = A.rowptr[i];
    for (long long j = 0; j < A.nnz; ++j) {
      h_colidx(j) = A.colidx[j];
      h_vals(j) = static_cast<MatT>(static_cast<float>(A.vals[j]));
    }
    for (int i = 0; i < A.n; ++i)
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
      const int rows_per_team = ts;  // one row per team thread
      const int leagues = (A.n + rows_per_team - 1) / rows_per_team;
      SpMV<MatT, VecT> op{rowptr, colidx, vals, x, y, A.n, rows_per_team};
      // Explicit team size and vector length: Kokkos' AUTO misbehaves here.
      typename SpMV<MatT, VecT>::team_policy policy(leagues, ts, vl);

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
            "checksum", Kokkos::RangePolicy<Device>(0, A.n),
            KOKKOS_LAMBDA(const int i, double &acc) { acc += y(i); }, sum);
        best = Timing{s, sum, ts, vl};
      }
    }
  return best;
}

template <class MatT, class VecT>
static void report(const HostCSR &A, int reps, double baseline_seconds) {
  const Timing t = time_spmv<MatT, VecT>(A, reps);
  const double bytes = model_bytes(A, stored_bytes<MatT>(), stored_bytes<VecT>());
  const double base_bytes = model_bytes(A, sizeof(double), sizeof(double));
  std::printf("%-12s %-5s %-5s %9.3f %8.1f %8.2fx %8.2fx %5d %5d %14.6e\n",
              A.name.c_str(), TypeName<MatT>::value, TypeName<VecT>::value,
              t.seconds * 1e3, bytes / t.seconds / 1e9, baseline_seconds / t.seconds,
              base_bytes / bytes, t.team_size, t.vector_length, t.checksum);
  std::fflush(stdout);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int reps = 20;
    std::vector<std::pair<int, int>> grids = {{7, 200}, {27, 96}};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
      if (a == "--small") grids = {{7, 100}, {27, 48}};
    }

    std::printf("# SpMV storage-format sweep; accumulator and output always fp64\n");
    std::printf("# exec space: %s\n", Device::name());
    std::printf("# stored widths (bytes): fp64=%zu fp32=%zu bf16=%zu fp16=%zu\n",
                stored_bytes<double>(), stored_bytes<float>(), stored_bytes<bhalf_t>(),
                stored_bytes<half_t>());
    std::printf("# 'predict' is the minimal-traffic byte ratio vs fp64/fp64;"
                " index traffic is why it is not 2x\n");
    std::printf("%-12s %-5s %-5s %9s %8s %9s %9s %5s %5s %14s\n", "matrix", "mat",
                "vec", "ms", "GB/s", "speedup", "predict", "team", "vlen", "checksum");

    for (const auto &g : grids) {
      const HostCSR A = stencil3d(g.second, g.first);
      std::printf("# %s: n = %d, nnz = %lld, nnz/row = %.1f\n", A.name.c_str(), A.n,
                  A.nnz, double(A.nnz) / A.n);
      const double base = time_spmv<double, double>(A, reps).seconds;
      report<double, double>(A, reps, base);
      report<float, double>(A, reps, base);
      report<float, float>(A, reps, base);
      report<bhalf_t, double>(A, reps, base);
      report<bhalf_t, bhalf_t>(A, reps, base);
      report<half_t, double>(A, reps, base);
      report<half_t, half_t>(A, reps, base);
    }
  }
  Kokkos::finalize();
  return 0;
}
