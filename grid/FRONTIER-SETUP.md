# Grid on Frontier: Peter's setup, and where to start measuring

Written 2026-10-06 from Grid `develop` `3232b438` (cloned at `~/Codes/Grid`).
Audience: Mark and later Claude sessions. Every path and number below is
from the repo; anything else is marked as an estimate or a question.

## How Peter works

- **Claude Code, with its notes committed.** `CLAUDE.md` at the repo root points
  to `skills/`: eight playbooks, one of which (`multigrid-design-notes.md`) is a
  Claude memory file committed as-is. 206 of 207 `develop` commits since June
  2026 are his, one titled "Claude's reorg to clean up parameters".
- **Job scripts hold the reasoning.** `systems/Frontier/*.job` carry long
  comments: what each cell tests and what to read in its output. Each run is a
  `run_cell` function that greps fault strings and certificate lines
  (`GALERKIN CERTIFICATE`, `IMPORT CERTIFICATE`, `VERIFY`, `FINAL Nrhs`) out of
  per-rank stdout.
- **Measurement is timeline plus model, with no hardware counters.**
    - roctx ranges (`GRID_TRACE`, `--enable-tracing=roctx`) and a
      `rocprof --sys-trace` line, commented out in `systems/Frontier/wrap.sh`.
    - Grid's own timers and the `--log Performance` channel.
    - GB/s computed from counted bytes over time.
    - A/B parameter sweeps.
  Nothing in the repo uses rocprof-compute, `--pmc` counters or ISA.

## The production run (`systems/Frontier/pvdagm_multigrid.job`)

| | |
|---|---|
| Binary | `examples/Example_pvdagm_multigrid` (PVdagM, 3 levels, dense bottom) |
| Lattice | 48^3x96, Ls 24, Möbius b=1.5 c=0.5, M5 1.8, mass 0.00078 |
| Ranks | 36 nodes, 288 GCDs, `--mpi 3.6.4.4` -> **16x8x12x24 x Ls24 per GCD** |
| Options | `--accelerator-threads 8 --shm 4096 --shm-mpi 0 --device-mem 40000 --comms-overlap` |
| Solver | Nrhs 12, nbasis 60, Block1 2.2.3.3, Block2 4.4.2.4, outer tol 1e-8 |
| Inputs | `CONFIG=/ccs/home/paboyle/configs/ckpoint_lat.1000`, `SUBSPACE=/lustre/orion/phy157/proj-shared/phy157_dwf/paboyle/subspace_nb64.scidac` |
| Logs | `--debug-stdout`, per rank under `$GRID_STDOUT_ROOT`; `--log Error,Warning,Message,Performance` |

Precision variants (`pvdagm_mixed_precision.job`):
- `<FinePrecision>` fp64/fp32: run time, fine level inside the preconditioner.
- Separate binaries `Example_pvdagm_multigrid_fp32coarse` and `..._fp32coarse_fp32dense`.
- `FineSloppyComms`: the halo is sent as fp32 on an fp64 operator, bf16 on an fp32 operator.

## Build and environment

From `systems/Frontier/sourceme-rocm7.2.sh` and `config-command`:

```bash
. <spack>/share/spack/setup-env.sh      # Peter's: /autofs/nccs-svm1_home1/paboyle/spack-frontier
export CLIME=... MPFR=... OPENSSL=... GMP=...   # from `spack find --paths`
module load cce/21.0.0 cpe/26.03 rocm/7.2.0 libfabric
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib/llvm/lib/:$LD_LIBRARY_PATH:$CRAY_LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=kdreg2 FI_HMEM_ROCR_USE_DMABUF=0   # libfabric faults, see systems/WorkArounds.txt
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_SMP_SINGLE_COPY_MODE=CMA MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=7 GRID_ALLOC_NCACHE_LARGE=0

../../configure --enable-comms=mpi-auto --with-lime=$CLIME --enable-unified=no \
  --enable-shm=nvlink --enable-tracing=roctx --enable-accelerator=hip \
  --enable-gen-simd-width=64 --disable-gparity --disable-fermion-reps --enable-simd=GPU \
  --with-openssl=$OPENSSL --with-gmp=$GMP --with-mpfr=$MPFR CXX=hipcc MPICXX=mpicxx \
  CXXFLAGS="-fPIC -I${ROCM_PATH}/include/ -I${MPICH_DIR}/include " \
  LDFLAGS="-L${ROCM_PATH}/lib -L${MPICH_DIR}/lib -lmpi -lmpi_gtl_hsa -lhipblas -lrocsolver -lrocblas -lhipfft -lamdhip64"
```

Rank-to-GPU binding (`select_gpu` in the job): `GPU_MAP=(0 1 2 3 7 6 5 4)`,
`NUMA_MAP=(3 3 1 1 2 2 0 0)`, `HIP_VISIBLE_DEVICES=$GPU`, `numactl -m $NUMA -N $NUMA`.

## What he has measured on the fine level

From `skills/multigrid-design-notes.md`, roctx trace, one Möbius `M` in the MG
smoother, 288 GCDs:

| | ms |
|---|---|
| whole `M` | 7.1 |
| `DhopInternalOverlappedComms` (wall) | 5.5 |
| `hipStreamSynchronize` x7 | 2.2 |
| Dslash interior + exterior kernels | 1.4 + 0.4 |
| two `M5D`, axpy | 0.5 each, 0.4 |

- About 63% of `M` has MPI in flight. Compute and communication are serialized
  despite `--comms-overlap`. His model: 4.7 ms of communication against 3.1 ms of
  work that could overlap it.
- `Benchmark_dwf`: fp32 sloppy comms gives +59%. Accelerator threads are
  second-order.
- Coarse MVM: 1119.6 GB/s (70% of HBM), with MPI latency 37% of each call.

This fits "80% of the time on the fine grid": the fine level is dominated by
communication and synchronization, not arithmetic.

## Where hardware counters come in

**Estimate, to be checked:** a full-lattice fp64 Dslash on 16x8x12x24x24 =
884,736 5D sites must move at least about 432 B per site (spinor in and out,
links shared across Ls), which is 382 MB. With no neighbour reuse it moves about
1.8 KB per site, 1.57 GB. At an achievable ~1.3 TB/s that is 0.3–1.2 ms, against
1.8 ms measured. So the kernels sit somewhere between 1.5x and 6x off the
bandwidth bound. Counters settle where: HBM bytes read and written, against the
382 MB floor. Taking time from the trace and bytes from the counters follows the
house rule (counter-tool durations are replay artifacts).

**Smallest faithful test:** the per-GCD problem is what the kernels see, so
reproduce it on one node:

```bash
srun -N1 -n8 ./select_gpu benchmarks/Benchmark_dwf --grid 16.16.24.48 --mpi 1.2.2.2 -Ls 24 \
     --accelerator-threads 8 --shm 4096 --comms-overlap
```

That gives 16x8x12x24 per GCD, the same as production. Note `-Ls` takes a
single dash, and the benchmark defaults to Ls 16 (Peter's trap note). The kernels
match production; on-node communication does not, so take overlap questions to
2 nodes or more.

## First measurements, in order

1. Build check plus `Benchmark_dwf` on 1 node at the production local volume.
   Confirm the trace numbers above (interior and exterior kernel times).
2. rocprof-compute on the Dslash interior, exterior and `M5D` kernels only
   (`-k`, `--no-roof` first). Record HBM bytes, L2 hit rate, occupancy and
   stall reasons.
3. Timeline on 2 nodes (roctx plus `rocprofv3 --sys-trace`). Find what the 7
   `hipStreamSynchronize` calls wait for, and why overlap fails.
4. The same at fp32 (`FinePrecision`) and with sloppy comms, where it matters
   for the MG preconditioner.

## Questions for Peter

- Read access to `phy157` proj-shared (configurations, subspace, his build), or
  a copy into our project?
- Can we use his spack environment, or should we build our own?
- A configuration that runs in about a minute on 1–4 nodes, for the smaller test?
- Do his roctx traces cover the 288-GCD run? Can we have the raw files?
